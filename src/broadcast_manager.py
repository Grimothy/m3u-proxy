"""
Network Broadcast Manager for m3u-proxy.

Manages FFmpeg processes for network broadcasting with:
- Duration-limited streaming for programme boundaries
- Segment sequence continuity across transitions
- Discontinuity marker support
- Webhook callbacks to Laravel when programmes end
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import httpx

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class BroadcastConfig:
    """Configuration for a network broadcast."""

    network_id: str
    stream_url: str
    seek_seconds: int = 0
    duration_seconds: int = 0  # 0 = unlimited
    segment_start_number: int = 0
    add_discontinuity: bool = False
    segment_duration: int = 6
    hls_list_size: int = 20
    transcode: bool = False
    video_bitrate: Optional[str] = None
    audio_bitrate: int = 192
    video_resolution: Optional[str] = None
    # Optional explicit codec/preset/hwaccel options (populated from Network transcode config)
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    preset: Optional[str] = None
    hwaccel: Optional[str] = None
    # Explicit audio stream index resolved by Laravel from the media server's own
    # metadata for a preferred-language selection. When set, this specific stream is
    # mapped instead of the first audio stream (index 0).
    audio_stream_index: Optional[int] = None
    # When set, detect an embedded subtitle track on the source and expose it as a
    # toggleable WebVTT rendition in the HLS output (master + subtitle variant
    # playlist), rather than burning it into the video.
    subtitles_enabled: bool = False
    # Explicit subtitle URL resolved by Laravel from the media server's own metadata
    # (covers embedded AND external/sidecar-file subtitles, which a raw ffprobe of the
    # video file can never see). When present, this is used directly as a second FFmpeg
    # input instead of probing the raw video stream for subtitles.
    subtitle_url: Optional[str] = None
    subtitle_language: Optional[str] = None
    # Seek offset the proxy must apply to the subtitle_url input.
    #   0    -> the subtitle URL was already seeked server-side (e.g. Emby's
    #           startPositionTicks path segment rebased the cues to zero at the same
    #           content-time the video was seeked to). The subtitle already shares the
    #           video's timeline origin, so the proxy adds no -ss/-itsoffset. PREFERRED.
    #   > 0  -> a full-file subtitle that could not be seeked server-side; the proxy seeks
    #           it locally with -ss and corrects rebasing with -itsoffset. FALLBACK.
    # Falls back to seek_seconds when not set, for backward compatibility.
    subtitle_seek_seconds: Optional[int] = None
    callback_url: Optional[str] = None
    # Optional custom headers to include when FFmpeg fetches the input URL
    headers: Optional[Dict[str, str]] = None
    # DVR mode: preserve all HLS segments (no rolling deletion) for post-processing
    dvr_mode: bool = False
    metadata: Optional[Dict] = None
    # Pre-queued next programme config for zero-round-trip auto-transition.
    # When FFmpeg exits with code 0, the process immediately starts this config
    # instead of waiting for a Laravel callback → start round-trip.
    next_stream_config: Optional["BroadcastConfig"] = None


@dataclass
class BroadcastStatus:
    """Status of a running broadcast."""

    network_id: str
    status: str  # starting, running, stopping, stopped, failed
    current_segment_number: int
    started_at: Optional[str]
    stream_url: str
    hls_dir: Optional[str] = None
    ffmpeg_pid: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None
    bytes_written: int = 0


class NetworkBroadcastProcess:
    """
    Manages a single network broadcast FFmpeg process.

    Key features:
    - Duration limiting via -t flag for programme boundaries
    - Segment number continuity via -start_number
    - Discontinuity injection via HLS flags
    - Webhook callback when FFmpeg exits
    """

    # Error patterns to detect in FFmpeg stderr
    INPUT_ERROR_PATTERNS = [
        "error opening input",
        "failed to resolve hostname",
        "connection refused",
        "connection timed out",
        "server returned 4",  # 403, 404, etc.
        "server returned 5",  # 500, 502, etc.
        "invalid data found",
        "no such file or directory",
        "protocol not found",
    ]

    # Patterns that match INPUT_ERROR_PATTERNS but are non-fatal — log as warning and continue.
    # e.g. FFmpeg's HLS muxer emits "failed to delete old segment" when it can't remove a
    # segment that was already cleaned up externally; this should never kill the broadcast.
    INPUT_ERROR_SUPPRESSIONS = [
        "failed to delete old segment",
    ]

    def __init__(self, config: BroadcastConfig, hls_base_dir: str):
        self.config = config
        self.network_id = config.network_id
        self.hls_dir = os.path.join(hls_base_dir, f"broadcast_{config.network_id}")
        self.process: Optional[asyncio.subprocess.Process] = None
        self.status = "starting"
        self.current_segment_number = config.segment_start_number
        self.started_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._stopping = False
        self._bytes_written: int = 0  # Cumulative bytes across all segments ever seen
        # Segment filenames already counted
        self._seen_segments: Set[str] = set()
        # Populated by _resolve_subtitle_state() before the command is built.
        # None = no subtitle stream detected (or subtitles_enabled is False) —
        # the command falls back to the original single-playlist output.
        # "" or a language code = a subtitle stream is present and mapped.
        self._subtitle_language: Optional[str] = None
        # Which FFmpeg input the mapped subtitle stream comes from: 0 = muxed into
        # the main video file, 1 = a separate explicit subtitle_url input.
        self._subtitle_input_index: int = 0
        # Cached result of the video-first-PTS probe (seconds). None = not yet probed.
        # Used to compute subtitle drift correction after a #range= byte-seek lands
        # mid-GOP; the probe is only paid for once per BroadcastProcess instance.
        self._cached_video_first_pts: Optional[float] = None

    async def _resolve_subtitle_state(self) -> None:
        """
        Determine subtitle availability for the current config, preferring an explicit
        subtitle_url (resolved by Laravel from the media server's own metadata — covers
        embedded AND external/sidecar-file subtitles) over probing the raw video stream.
        Falls back to ffprobe self-detection only when no explicit URL was provided,
        which is the only option for sources without a media-server API (Local/WebDAV).
        """
        if self.config.subtitle_url:
            if await self._subtitle_url_has_content(self.config.subtitle_url):
                self._subtitle_language = self.config.subtitle_language or ""
                self._subtitle_input_index = 1
                return

            logger.info(
                f"Broadcast {self.network_id}: subtitle_url returned no usable "
                "content (likely seeked past the end of the subtitle track); "
                "continuing without subtitles"
            )
            self._subtitle_language = None
            self._subtitle_input_index = 0
            return

        self._subtitle_input_index = 0
        if not self.config.subtitles_enabled:
            self._subtitle_language = None
            return

        self._subtitle_language = await self._probe_subtitle_language()
        if self._subtitle_language is None:
            logger.info(
                f"Broadcast {self.network_id}: subtitles_enabled but no subtitle "
                "stream detected on source; continuing without subtitles"
            )

    async def _subtitle_url_has_content(self, url: str, min_bytes: int = 10) -> bool:
        """
        Verify an explicit subtitle_url actually returns usable content before handing
        it to FFmpeg as an input. Media servers' server-side seeked subtitle endpoints
        (e.g. Emby/Jellyfin's startPositionTicks) can return HTTP 200 with a completely
        empty body when the seek position falls after the subtitle track's last cue —
        e.g. resuming a mid-programme broadcast past where dialogue/subtitles end, or a
        subtitle file that's simply shorter than the video. An empty file has no
        parseable format, so FFmpeg aborts input probing for THE WHOLE PROCESS (video
        and audio included), not just the subtitle stream. The optional map syntax
        (-map 1:s:0?) does not help here — it only tolerates a missing stream inside an
        otherwise-valid container, not an input file that fails to open at all.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                return response.status_code == 200 and len(response.content) >= min_bytes
        except Exception as e:
            logger.warning(
                f"Broadcast {self.network_id}: failed to validate subtitle_url, "
                f"continuing without subtitles: {e}"
            )
            return False

    # Bitmap/image-based subtitle codecs cannot be transcoded to WebVTT (a text
    # format) — FFmpeg refuses with "Subtitle encoding currently only possible
    # from text to text or bitmap to bitmap" and aborts the ENTIRE process
    # (video and audio included), not just the subtitle output. Only accept
    # text-based codecs here.
    _TEXT_SUBTITLE_CODECS = {
        "subrip",
        "srt",
        "ass",
        "ssa",
        "mov_text",
        "webvtt",
        "text",
        "ttml",
    }

    async def _probe_subtitle_language(self) -> Optional[str]:
        """
        Probe the source for a text-based subtitle stream, returning its language
        tag ("" if untagged), or None if no usable subtitle stream exists or the
        probe fails/times out.

        This MUST run before building the FFmpeg command: -var_stream_map
        referencing a subtitle output that doesn't exist aborts the entire
        FFmpeg process (video and audio included), not just the subtitle
        track. Failing closed (None) on any doubt is intentional.
        """
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "s",
                "-show_entries",
                "stream=index,codec_name:stream_tags=language",
                "-of",
                "json",
                self.config.stream_url,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning(
                    f"Broadcast {self.network_id}: subtitle probe timed out, continuing without subtitles"
                )
                return None

            if proc.returncode != 0:
                return None

            data = json.loads(stdout or b"{}")
            streams = data.get("streams") or []
            if not streams:
                return None

            # -map 0:s:0? always selects the FIRST subtitle stream in source order,
            # so it must be the first one we inspect here too.
            first = streams[0]
            codec_name = (first.get("codec_name") or "").lower()
            if codec_name not in self._TEXT_SUBTITLE_CODECS:
                logger.info(
                    f"Broadcast {self.network_id}: first subtitle stream is "
                    f"'{codec_name}' (not text-based), continuing without subtitles"
                )
                return None

            language = (first.get("tags") or {}).get("language", "") or ""
            # Sanitize: this value is interpolated into the -var_stream_map argument,
            # which is comma/colon-delimited. Keep it to safe identifier characters.
            language = re.sub(r"[^a-zA-Z0-9_-]", "", language)[:10]
            return language
        except Exception as e:
            logger.warning(
                f"Broadcast {self.network_id}: subtitle probe failed, continuing without subtitles: {e}"
            )
            return None

    def _probe_video_first_pts(self) -> float:
        """
        Return the first video packet's PTS (in seconds) when reading from the main
        stream_url. Used to compute subtitle drift correction after a #range= byte-seek
        lands mid-GOP.

        After a byte-seek to offset N, ffmpeg waits for the next GOP boundary before
        emitting the first video packet, so the first PTS is typically 1-3s (the GOP
        size) rather than 0. The subtitle URL was server-pre-seeked to exact PTS 0 by
        Emby's startPositionTicks, so its cues land ahead of the video without
        correction. This probe runs once per BroadcastProcess instance; subsequent calls
        return the cached value.

        Returns 0.0 on any failure (probe timeout, ffprobe not installed, network
        error, parse error) so the caller can skip the correction without crashing.
        """
        if self._cached_video_first_pts is not None:
            return self._cached_video_first_pts

        try:
            import tempfile
            import os
            # ffprobe returns "N/A" for direct HTTP sources (no per-packet PTS
            # metadata available before ffmpeg has actually read some packets),
            # so we must first capture a tiny TS chunk to disk and probe THAT
            # file instead. ~0.5s of stream is enough for the first GOP-boundary
            # frame to land; the whole probe takes ~0.4s on a fast network.
            with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", self.config.stream_url,
                    "-t", "0.5",
                    "-c", "copy",
                    "-f", "mpegts",
                    tmp_path,
                ]
                subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=10)
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-show_entries", "packet=pts_time",
                        "-select_streams", "v",
                        "-of", "csv=p=0",
                        tmp_path,
                    ],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout:
                    pts_str = result.stdout.strip().split("\n")[0].rstrip(",")
                    if pts_str in ("N/A", "", "nan", "NaN"):
                        logger.debug(
                            f"Broadcast {self.network_id}: ffprobe returned no PTS "
                            f"for first video packet ('{pts_str}'); skipping drift correction"
                        )
                        self._cached_video_first_pts = 0.0
                        return 0.0
                    pts = float(pts_str)
                    self._cached_video_first_pts = pts
                    logger.info(
                        f"Broadcast {self.network_id}: video first PTS = {pts:.3f}s "
                        f"(after #range= byte-seek; will apply -itsoffset +{pts:.3f} on subtitle input)"
                    )
                    return pts
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Broadcast {self.network_id}: video PTS probe timed out after 10s; "
                f"skipping subtitle drift correction"
            )
        except FileNotFoundError:
            logger.warning(
                f"Broadcast {self.network_id}: ffprobe not found; "
                f"skipping subtitle drift correction"
            )
        except Exception as e:
            logger.warning(
                f"Broadcast {self.network_id}: video PTS probe failed: {e}; "
                f"skipping subtitle drift correction"
            )

        self._cached_video_first_pts = 0.0
        return 0.0

    def _build_ffmpeg_command(self) -> List[str]:
        """Build the FFmpeg command for HLS broadcast output."""
        cmd = ["ffmpeg", "-y"]

        # Hardware acceleration for DECODING must come BEFORE -i (input options)
        # Only add if it's a valid value (not None, empty, or "none")
        if self.config.transcode:
            hwaccel = getattr(self.config, "hwaccel", None)
            if hwaccel and hwaccel.lower() not in ("none", ""):
                cmd.extend(["-hwaccel", hwaccel])

        # Input-level seeking (BEFORE -i for accuracy)
        if self.config.seek_seconds > 0:
            cmd.extend(["-ss", str(self.config.seek_seconds)])

        # Real-time pacing - critical for live streaming
        cmd.append("-re")

        # Reconnection options for network streams
        cmd.extend(
            [
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "10",
            ]
        )

        # Input URL
        # If headers are provided explicitly in the BroadcastConfig, prefer them.
        if (
            getattr(self.config, "headers", None)
            and isinstance(self.config.headers, dict)
            and isinstance(self.config.stream_url, str)
            and (
                "://" in self.config.stream_url
                and not self.config.stream_url.startswith("file://")
            )
        ):
            try:
                headers = []
                for hk, hv in self.config.headers.items():
                    # sanitize header names/values
                    k = str(hk).replace("\r", "").replace("\n", "").strip()
                    v = str(hv).replace("\r", "").replace("\n", "").strip()
                    if not k:
                        continue
                    headers.append(f"{k}: {v}")

                if headers:
                    header_str = "\r\n".join(headers) + "\r\n"
                    cmd.extend(["-headers", header_str, "-i", self.config.stream_url])
                else:
                    cmd.extend(["-i", self.config.stream_url])
            except Exception as e:
                logger.warning(f"Failed to construct headers for FFmpeg input: {e}")
                cmd.extend(["-i", self.config.stream_url])

        # If no input has been added by the header logic above, append it as a plain -i.
        # This is a defensive measure to avoid malformed commands when headers are not
        # provided and the URL is a simple network resource (e.g., Emby/Jellyfin stream.ts).
        if "-i" not in cmd:
            cmd.extend(["-i", self.config.stream_url])

        # Subtitles are active either because an explicit subtitle_url was resolved by
        # Laravel (input index 1 — a separate file, e.g. an external/sidecar subtitle) or
        # because _probe_subtitle_language() found one muxed into the main video (input
        # index 0). Requesting a subtitle output that doesn't exist aborts the whole
        # process below via -var_stream_map, so this must never be assumed — only
        # resolved/probed ahead of time by _resolve_subtitle_state().
        subtitles_active = self._subtitle_language is not None

        # A separate subtitle input MUST be added here — before -t below — so that
        # ffmpeg's per-input option scoping doesn't misattribute -t to this input
        # instead of applying it as the intended output-level duration limit.
        if subtitles_active and self._subtitle_input_index == 1:
            # Seek to match the main input's effective playback position so subtitle cue
            # timestamps stay in sync with a resumed/seeked broadcast instead of restarting
            # from the subtitle file's own time zero.
            #
            # PREFERRED PATH (subtitle_seek_seconds == 0): Laravel now asks the media server
            # to seek the subtitle URL server-side (e.g. Emby's startPositionTicks path
            # segment), which rebases the cues to zero at the same content-time the video was
            # seeked to. The subtitle then already shares the video's timeline origin, so we
            # add NO -ss / -itsoffset here — a single seek authority (the media server) drives
            # both streams, which is what keeps them frame-locked. This is the branch that
            # runs for Emby/Jellyfin broadcasts.
            #
            # EXPLICIT PATH (subtitle_seek_seconds > 0): a full-file subtitle that could not
            # be seeked server-side. We seek it locally with -ss and correct the rebasing with
            # -itsoffset (see below). PHP always sends this explicitly after the Jul 6
            # single-authority fix; it relies on demuxer-specific rebasing behavior and is
            # inherently more fragile than the server-side path.
            #
            # FALLBACK (subtitle_seek_seconds is None): older clients may not send
            # subtitle_seek_seconds. Fall back to 0 (no seek) — NEVER to seek_seconds,
            # which is the VIDEO input's seek and would push subtitle cue timestamps out
            # of sync with the video.
            subtitle_seek = self.config.subtitle_seek_seconds or 0
            if subtitle_seek > 0:
                cmd.extend(["-ss", str(subtitle_seek)])
                # -ss skips the subtitle file's early cues but does NOT rebase the
                # timestamps of the ones that survive — they stay absolute (relative to
                # the subtitle file's own time zero). Emby/Jellyfin's VideoCodec=copy
                # remux, however, DOES rebase the video's own PTS to ~0 at the seek
                # point (confirmed via ffprobe on a live source: first video packet PTS
                # ≈ 0, not ≈ the seek offset). Without correcting for that mismatch,
                # every surviving subtitle cue is subtitle_seek seconds too late
                # relative to the video it's supposed to caption. -itsoffset shifts this
                # input's output timestamps by the same (negative) amount to re-align it
                # with the video's rebased timeline.
                cmd.extend(["-itsoffset", str(-subtitle_seek)])
            # Without reconnect options, Emby/Jellyfin closing this connection mid-broadcast
            # (observed via a lingering CLOSE-WAIT socket on a multi-hour live run) leaves
            # ffmpeg with a dead subtitle input it never retries. Because the subtitle map
            # below is optional (`?`), ffmpeg doesn't error out — it just silently stops
            # emitting any further subtitle cues for the rest of the broadcast, which looks
            # like subtitles "going out of sync" or disappearing rather than a crash.
            cmd.extend(
                [
                    "-reconnect",
                    "1",
                    "-reconnect_streamed",
                    "1",
                    "-reconnect_delay_max",
                    "10",
                ]
            )

            # Drift correction: after a #range= byte-seek on the video input, ffmpeg waits
            # for the next GOP boundary before emitting the first packet, so the video's
            # first PTS is offset by the GOP size (~1-3s typically). The subtitle input
            # was server-pre-seeked by Emby to exact PTS 0, so its cues land ahead of the
            # video. Shift the subtitle's output timestamps forward by the video's first
            # PTS so cues land aligned with the video's GOP start.
            # Only probe when seek_seconds > 0 (the byte-seek path); when seek_seconds=0
            # the video starts at PTS 0 so there is no drift to correct.
            if self.config.seek_seconds > 0:
                video_drift = self._probe_video_first_pts()
                if video_drift > 0.1:  # only apply when drift is meaningful (>100ms)
                    cmd.extend(["-itsoffset", str(video_drift)])

            cmd.extend(["-i", self.config.subtitle_url])

        # Duration limiting for programme boundary
        if self.config.duration_seconds > 0:
            cmd.extend(["-t", str(self.config.duration_seconds)])

        # Stream mapping - video + audio (+ subtitle, if detected). Video and audio are
        # optional to support audio-only streams (e.g. radio stations) and video-only
        # streams. FFmpeg silently skips missing optional streams. An explicit
        # audio_stream_index (resolved by Laravel for a preferred-language selection)
        # maps that specific stream instead of the default first audio stream.
        #
        # Laravel resolves audio_stream_index from the media server's own MediaStreams
        # metadata, which is an ABSOLUTE index spanning video+audio+subtitle streams
        # together (e.g. video=0, audio=1, subtitles=2..N). FFmpeg's "a:N" specifier is
        # type-RELATIVE (the Nth audio stream) — those only coincide when audio happens
        # to be the very first stream in the file, which is virtually never true. Using
        # "a:N" here silently maps nothing whenever N doesn't match an actual audio-type
        # position (the "?" swallows the miss), which then makes "-var_stream_map"'s
        # "a:0" reference dangling and aborts the WHOLE HLS conversion — not just audio.
        # The absolute-index form ("0:N", no type letter) selects by the same numbering
        # Laravel resolved against, so it always lands on the right stream.
        audio_map = (
            f"0:{self.config.audio_stream_index}?"
            if self.config.audio_stream_index is not None
            else "0:a:0?"
        )
        cmd.extend(["-map", "0:v:0?", "-map", audio_map])
        if subtitles_active:
            cmd.extend(["-map", f"{self._subtitle_input_index}:s:0?"])

        # Codec selection
        if self.config.transcode:
            # Video codec selection (allow explicit codec like libx264 or h264_nvenc)
            # Only applied when a video stream is actually present (0:v:0? maps nothing for audio-only)
            video_codec = self.config.video_codec or "libx264"
            cmd.extend(["-c:v", video_codec])

            # Preset - default to 'veryfast' for real-time encoding if not specified
            # This is critical for avoiding encoding bottlenecks that cause audio drift
            preset = getattr(self.config, "preset", None) or "veryfast"
            cmd.extend(["-preset", preset])

            if self.config.video_bitrate:
                cmd.extend(["-b:v", f"{self.config.video_bitrate}k"])
            if self.config.video_resolution:
                cmd.extend(["-vf", f"scale={self.config.video_resolution}"])

            # Audio codec and bitrate
            audio_codec = self.config.audio_codec or "aac"
            cmd.extend(["-c:a", audio_codec, "-b:a", f"{self.config.audio_bitrate}k"])

            # Force standard broadcast audio settings to prevent sample rate mismatches
            # that cause "deep/slow" audio playback issues
            cmd.extend(["-ar", "48000"])  # 48kHz is standard for broadcast
            cmd.extend(["-ac", "2"])  # Force stereo output
        else:
            cmd.extend(["-c:v", "copy", "-c:a", "copy"])

        if subtitles_active:
            cmd.extend(["-c:s", "webvtt"])
            # HLS's #EXT-X-STREAM-INF requires a BANDWIDTH value, which FFmpeg only
            # emits when it has an explicit bitrate for the stream. In copy mode
            # (and whenever video_bitrate/audio_bitrate weren't already added above)
            # there's no encoder bitrate to read, so add a hint purely for the
            # manifest — it does not affect actual stream quality since -c:v/-c:a
            # copy ignores -b:v/-b:a for encoding.
            if "-b:v" not in cmd:
                cmd.extend(["-b:v", f"{self.config.video_bitrate or 2000}k"])
            if "-b:a" not in cmd:
                cmd.extend(["-b:a", f"{self.config.audio_bitrate}k"])

        # HLS output configuration
        cmd.extend(["-f", "hls"])
        cmd.extend(["-hls_time", str(self.config.segment_duration)])
        # DVR mode: hls_list_size=0 keeps all segments in the manifest for concat
        hls_list_size = 0 if self.config.dvr_mode else self.config.hls_list_size
        cmd.extend(["-hls_list_size", str(hls_list_size)])
        cmd.extend(["-start_number", str(self.config.segment_start_number)])

        # HLS flags — DVR mode keeps all segments for post-processing concat
        hls_flags = [
            "program_date_time",
            "omit_endlist",
            "independent_segments",
        ]
        if not self.config.dvr_mode:
            # Rolling-window live broadcasts delete old segments to save space
            hls_flags.insert(0, "delete_segments")
        if self.config.add_discontinuity:
            hls_flags.append("discont_start")
        cmd.extend(["-hls_flags", "+".join(hls_flags)])

        if subtitles_active:
            # A subtitle rendition forces FFmpeg's variant-stream mode: instead of one
            # flat live.m3u8, it emits a master playlist referencing a video variant
            # playlist and a subtitle variant playlist (numeric %v, no `name:` — that
            # would substitute a string into filenames and complicate serving them).
            var_stream_map = "v:0,a:0,s:0,sgroup:subs"
            if self._subtitle_language:
                var_stream_map += f",language:{self._subtitle_language}"
            cmd.extend(["-var_stream_map", var_stream_map])
            cmd.extend(["-master_pl_name", "master.m3u8"])

            segment_pattern = os.path.join(self.hls_dir, "live%v_%06d.ts")
            cmd.extend(["-hls_segment_filename", segment_pattern])

            playlist_path = os.path.join(self.hls_dir, "live_%v.m3u8")
            cmd.append(playlist_path)
        else:
            # Segment filename template (6-digit zero-padded)
            segment_pattern = os.path.join(self.hls_dir, "live%06d.ts")
            cmd.extend(["-hls_segment_filename", segment_pattern])

            # Output playlist
            playlist_path = os.path.join(self.hls_dir, "live.m3u8")
            cmd.append(playlist_path)

        return cmd

    async def start(self) -> bool:
        """Start the FFmpeg broadcast process."""
        try:
            # Ensure HLS directory exists with proper permissions
            os.makedirs(self.hls_dir, exist_ok=True)
            try:
                os.chmod(self.hls_dir, 0o755)
            except Exception as e:
                logger.warning(f"Failed to set permissions on {self.hls_dir}: {e}")

            # On a fresh start (not a transition), remove any leftover segments/playlists
            # so FFmpeg's rolling-window deletion doesn't hit files it didn't write.
            if (
                self.config.segment_start_number == 0
                and not self.config.add_discontinuity
            ):
                stale_count = 0
                for filename in (
                    list(os.listdir(self.hls_dir))
                    if os.path.isdir(self.hls_dir)
                    else []
                ):
                    if filename.endswith((".ts", ".m3u8", ".vtt")):
                        try:
                            os.remove(os.path.join(self.hls_dir, filename))
                            stale_count += 1
                        except FileNotFoundError:
                            pass
                        except OSError as e:
                            logger.warning(
                                f"Broadcast {self.network_id}: could not remove stale file {filename}: {e}"
                            )
                if stale_count:
                    logger.info(
                        f"Broadcast {self.network_id}: removed {stale_count} stale file(s) before fresh start"
                    )

            await self._resolve_subtitle_state()

            cmd = self._build_ffmpeg_command()
            logger.info(f"Starting broadcast {self.network_id}: {' '.join(cmd)}")

            self.process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
            )

            self.started_at = datetime.now(timezone.utc)
            self.status = "running"

            # Start monitoring tasks
            self._stderr_task = asyncio.create_task(self._log_stderr())
            self._monitor_task = asyncio.create_task(self._monitor_process())
            self._poll_task = asyncio.create_task(self._poll_bytes())

            logger.info(
                f"Broadcast {self.network_id} started with PID {self.process.pid}"
            )
            return True

        except Exception as e:
            self.status = "failed"
            self.error_message = str(e)
            logger.error(f"Failed to start broadcast {self.network_id}: {e}")
            return False

    async def stop(self, graceful: bool = True) -> int:
        """
        Stop the FFmpeg process.

        Args:
            graceful: If True, send SIGTERM and wait; if False, send SIGKILL immediately.

        Returns:
            The final segment number.
        """
        self._stopping = True
        self.status = "stopping"

        if self.process and self.process.returncode is None:
            try:
                if graceful:
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Broadcast {self.network_id} did not terminate gracefully, killing"
                        )
                        self.process.kill()
                        await self.process.wait()
                else:
                    self.process.kill()
                    await self.process.wait()
            except ProcessLookupError:
                pass  # Process already dead

        # Cancel monitoring tasks
        for task in [self._monitor_task, self._stderr_task, self._poll_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Get final segment number from files
        final_segment = self._get_final_segment_number()
        self.current_segment_number = final_segment
        self.status = "stopped"

        logger.info(
            f"Broadcast {self.network_id} stopped, final segment: {final_segment}"
        )
        return final_segment

    # Patterns to skip in FFmpeg output (verbose/noisy messages)
    SKIP_LOG_PATTERNS = [
        "frame=",  # Progress output
        "fps=",  # FPS stats
        "time=",  # Time stats
        "bitrate=",  # Bitrate stats
        "speed=",  # Speed stats
        # Size stats (N/A for HLS stream-copy; tracked via segment polling)
        "size=",
        "resumed reading",  # Reconnection noise
        "opening",  # File opening messages (lowercase)
        "muxing overhead",  # Summary stats
        "video:",  # Summary stats
        "audio:",  # Summary stats
    ]

    async def _log_stderr(self):
        """Monitor FFmpeg stderr for errors only. Suppresses verbose output."""
        if not self.process or not self.process.stderr:
            return

        buf = b""
        try:
            while self.process.returncode is None:
                chunk = await self.process.stderr.read(4096)
                if not chunk:
                    break

                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if not line_str:
                        continue

                    line_lower = line_str.lower()

                    # Check for input errors
                    is_input_error = any(
                        p in line_lower for p in self.INPUT_ERROR_PATTERNS
                    )
                    if is_input_error:
                        # Some error patterns are non-fatal (e.g. segment already deleted)
                        is_suppressed = any(
                            p in line_lower for p in self.INPUT_ERROR_SUPPRESSIONS
                        )
                        if is_suppressed:
                            logger.warning(
                                f"Broadcast {self.network_id} (non-fatal): {line_str}"
                            )
                            continue

                        self.error_message = line_str
                        self.status = "failed"
                        logger.error(f"Broadcast {self.network_id} error: {line_str}")
                        await self._send_callback(
                            "broadcast_failed",
                            {"error": line_str, "error_type": "input_error"},
                        )
                        return

                    # Skip verbose/noisy messages entirely
                    should_skip = any(
                        pattern in line_lower for pattern in self.SKIP_LOG_PATTERNS
                    )
                    if should_skip:
                        continue

                    # Log warnings and errors only
                    if (
                        "error" in line_lower
                        or "warning" in line_lower
                        or "failed" in line_lower
                    ):
                        logger.warning(f"Broadcast {self.network_id}: {line_str}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error reading FFmpeg stderr for {self.network_id}: {e}")

    async def _monitor_process(self):
        """Monitor FFmpeg process and send callback when it exits."""
        if not self.process:
            return

        try:
            while True:
                await self.process.wait()

                # Skip callback on intentional stop — the editor initiated it and handles
                # post-processing directly without waiting for a proxy callback.
                if self._stopping:
                    return

                # Determine final segment number
                final_segment = self._get_final_segment_number()
                self.current_segment_number = final_segment

                # Calculate duration streamed
                duration_streamed = 0.0
                if self.started_at:
                    duration_streamed = (
                        datetime.now(timezone.utc) - self.started_at
                    ).total_seconds()

                exit_code = self.process.returncode

                if exit_code == 0 and self.config.next_stream_config:
                    # Auto-transition: immediately start the next programme without
                    # waiting for the Laravel callback → start round-trip.
                    next_config = self.config.next_stream_config
                    next_config.network_id = self.network_id
                    next_config.segment_start_number = final_segment + 1
                    next_config.add_discontinuity = True

                    # Cancel old stderr/poll tasks before swapping the subprocess.
                    for task in [self._stderr_task, self._poll_task]:
                        if task and not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass

                    # Reset per-segment tracking for the new programme.
                    self._bytes_written = 0
                    self._seen_segments = set()
                    self.error_message = None
                    self.config = next_config

                    # Re-resolve for the new programme's content — the previous
                    # programme's subtitle availability/language/URL does not carry over.
                    previous_subtitles_active = self._subtitle_language is not None
                    await self._resolve_subtitle_state()
                    new_subtitles_active = self._subtitle_language is not None

                    if previous_subtitles_active != new_subtitles_active:
                        # The manifest shape is changing (subtitled <-> non-subtitled).
                        # Remove the outgoing mode's manifest file(s) so
                        # get_playlist_path()/cleanup_orphaned_segments() never mistake
                        # a stale leftover for the current invocation's output. Segment
                        # files (.ts/.vtt) are left alone — only manifests are mode-specific.
                        stale_files = (
                            ["live.m3u8"]
                            if new_subtitles_active
                            else ["master.m3u8", "live_0.m3u8", "live_0_vtt.m3u8"]
                        )
                        for stale_name in stale_files:
                            try:
                                os.remove(os.path.join(self.hls_dir, stale_name))
                            except FileNotFoundError:
                                pass
                            except OSError as e:
                                logger.warning(
                                    f"Broadcast {self.network_id}: could not remove stale manifest {stale_name}: {e}"
                                )

                    logger.info(
                        f"Broadcast {self.network_id}: auto-transitioning from "
                        f"segment {final_segment} to next programme"
                    )

                    cmd = self._build_ffmpeg_command()
                    logger.info(
                        f"Broadcast {self.network_id}: starting next programme: "
                        + " ".join(cmd[:6])
                        + " ..."
                    )

                    try:
                        self.process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        self.started_at = datetime.now(timezone.utc)
                        self.status = "running"
                        new_pid = self.process.pid
                        logger.info(
                            f"Broadcast {self.network_id}: auto-transitioned, "
                            f"PID {new_pid}, segment start "
                            f"{next_config.segment_start_number}"
                        )
                    except Exception as exc:
                        logger.error(
                            f"Broadcast {self.network_id}: auto-transition "
                            f"failed to start FFmpeg: {exc}"
                        )
                        self.status = "failed"
                        self.error_message = str(exc)
                        new_pid = None

                    # Notify Laravel asynchronously — don't block the new process.
                    asyncio.create_task(
                        self._send_callback(
                            "programme_ended",
                            {
                                "exit_code": exit_code,
                                "final_segment_number": final_segment,
                                "duration_streamed": duration_streamed,
                                "auto_transitioned": True,
                                "new_pid": new_pid,
                            },
                        )
                    )

                    if new_pid:
                        # Restart per-process monitoring tasks and loop to watch new process.
                        self._stderr_task = asyncio.create_task(self._log_stderr())
                        self._poll_task = asyncio.create_task(self._poll_bytes())
                        continue
                    else:
                        return

                elif exit_code == 0:
                    # Normal completion (duration limit reached) or intentional DVR stop.
                    self.status = "stopped"
                    await self._send_callback(
                        "programme_ended",
                        {
                            "exit_code": exit_code,
                            "final_segment_number": final_segment,
                            "duration_streamed": duration_streamed,
                        },
                    )
                    break
                else:
                    # Abnormal exit
                    self.status = "failed"
                    self.error_message = f"FFmpeg exited with code {exit_code}"
                    await self._send_callback(
                        "broadcast_failed",
                        {
                            "exit_code": exit_code,
                            "final_segment_number": final_segment,
                            "duration_streamed": duration_streamed,
                            "error": self.error_message,
                        },
                    )
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error monitoring broadcast {self.network_id}: {e}")

    async def _send_callback(self, event: str, data: dict):
        """Send webhook callback to Laravel."""
        if not self.config.callback_url:
            logger.debug(
                f"No callback URL for broadcast {self.network_id}, skipping callback"
            )
            return

        payload = {
            "network_id": self.network_id,
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dvr_mode": self.config.dvr_mode,
            "hls_dir": self.hls_dir if self.config.dvr_mode else None,
            "data": data,
        }

        try:
            timeout = getattr(settings, "BROADCAST_CALLBACK_TIMEOUT", 10)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.config.callback_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "M3U-Proxy-Broadcast/1.0",
                    },
                )
                if response.status_code >= 400:
                    logger.warning(
                        f"Callback to {self.config.callback_url} failed with status {response.status_code}"
                    )
                else:
                    logger.info(
                        f"Callback sent for broadcast {self.network_id}: {event}"
                    )
        except Exception as e:
            logger.error(f"Error sending callback for broadcast {self.network_id}: {e}")

    async def _poll_bytes(self, interval: float = 1.0) -> None:
        """
        Track cumulative bytes by watching for new .ts segment files.

        FFmpeg's progress output reports ``size=N/A`` for HLS stream-copy output,
        so we can't use stderr parsing. Instead, we scan the HLS directory every
        ``interval`` seconds. Each new segment file we haven't seen before is
        measured and added to ``_bytes_written``.

        For DVR broadcasts the files are never deleted, so every segment is counted
        exactly once. For non-DVR broadcasts FFmpeg deletes old segments via its
        ``delete_segments`` flag, but the rolling window keeps several segments on
        disk at any time (hls_list_size × hls_time seconds), giving us a comfortable
        window to measure each file before it disappears.
        """
        try:
            while not self._stopping:
                try:
                    if os.path.exists(self.hls_dir):
                        for filename in os.listdir(self.hls_dir):
                            if (
                                filename.endswith(".ts")
                                and filename not in self._seen_segments
                            ):
                                self._seen_segments.add(filename)
                                try:
                                    self._bytes_written += os.path.getsize(
                                        os.path.join(self.hls_dir, filename)
                                    )
                                except OSError:
                                    pass
                except Exception:
                    pass
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    def _get_bytes_written(self) -> int:
        """Return cumulative bytes written across all segments ever seen."""
        return self._bytes_written

    def _get_final_segment_number(self) -> int:
        """Get the highest segment number from existing files."""
        try:
            if not os.path.exists(self.hls_dir):
                return self.config.segment_start_number

            # Pattern: live000001.ts -> extract 000001. When subtitles are active,
            # segments are named live{variant}_000001.ts (e.g. live0_000001.ts) —
            # the optional `\d+_` group tolerates that variant-index prefix.
            pattern = re.compile(r"live(?:\d+_)?(\d{6})\.ts$")
            max_segment = self.config.segment_start_number

            for filename in os.listdir(self.hls_dir):
                match = pattern.match(filename)
                if match:
                    segment_num = int(match.group(1))
                    max_segment = max(max_segment, segment_num)

            return max_segment
        except Exception as e:
            logger.error(
                f"Error getting final segment number for {self.network_id}: {e}"
            )
            return self.config.segment_start_number

    @staticmethod
    def parse_playlist_segments(playlist_path: str) -> Set[str]:
        """Return the set of .ts filenames referenced by an HLS playlist."""
        segments: Set[str] = set()
        try:
            with open(playlist_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        segments.add(os.path.basename(line))
        except Exception:
            pass
        return segments

    def cleanup_orphaned_segments(self, age_threshold: int = 0) -> int:
        """
        Remove .ts files in the HLS dir that are not referenced by the current playlist.

        Args:
            age_threshold: Only remove files older than this many seconds.
                           0 = remove immediately (used during programme transitions).

        Returns:
            Number of files removed.
        """
        # Whether subtitles are active for the CURRENT ffmpeg invocation. This must be
        # keyed off self._subtitle_language, not "does master.m3u8 exist on disk" — a
        # transition from subtitled to non-subtitled content (or vice versa) leaves the
        # previous invocation's manifest file(s) on disk (stale-file cleanup is skipped
        # during transitions to preserve segment continuity), which would otherwise make
        # this permanently misidentify the current mode after a mode-changing transition.
        subtitles_active = self._subtitle_language is not None
        if subtitles_active:
            # Referenced segments (both .ts and .vtt) are split across the video and
            # subtitle variant playlists rather than one flat live.m3u8.
            referenced = self.parse_playlist_segments(
                os.path.join(self.hls_dir, "live_0.m3u8")
            ) | self.parse_playlist_segments(
                os.path.join(self.hls_dir, "live_0_vtt.m3u8")
            )
        else:
            playlist_path = os.path.join(self.hls_dir, "live.m3u8")
            if not os.path.exists(playlist_path):
                return 0
            referenced = self.parse_playlist_segments(playlist_path)

        removed = 0
        now = time.time()

        try:
            for filename in os.listdir(self.hls_dir):
                if not filename.endswith((".ts", ".vtt")):
                    continue
                if filename in referenced:
                    continue

                full_path = os.path.join(self.hls_dir, filename)
                if age_threshold > 0:
                    try:
                        if now - os.path.getmtime(full_path) < age_threshold:
                            continue
                    except OSError:
                        continue

                try:
                    os.remove(full_path)
                    removed += 1
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.warning(
                        f"Broadcast {self.network_id}: could not remove orphaned segment {filename}: {e}"
                    )
        except Exception as e:
            logger.error(
                f"Broadcast {self.network_id}: error during orphan cleanup: {e}"
            )

        if removed:
            logger.info(
                f"Broadcast {self.network_id}: removed {removed} orphaned segment(s)"
            )
        return removed

    def get_status(self) -> BroadcastStatus:
        """Get current broadcast status."""
        return BroadcastStatus(
            network_id=self.network_id,
            status=self.status,
            current_segment_number=self._get_final_segment_number(),
            started_at=self.started_at.isoformat() if self.started_at else None,
            stream_url=self.config.stream_url,
            hls_dir=self.hls_dir,
            ffmpeg_pid=self.process.pid if self.process else None,
            error_message=self.error_message,
            metadata=self.config.metadata,
            bytes_written=self._get_bytes_written(),
        )

    def get_playlist_path(self) -> Optional[str]:
        """
        Get path to the HLS playlist file for the CURRENT ffmpeg invocation.

        When subtitles are active FFmpeg writes a master.m3u8 (referencing a video
        variant + subtitle variant playlist) instead of a flat live.m3u8. This is
        keyed off self._subtitle_language rather than "does master.m3u8 exist on
        disk" — a transition from subtitled to non-subtitled content (or vice versa)
        leaves the previous invocation's manifest file on disk (stale-file cleanup
        is skipped during transitions to preserve segment continuity), which would
        otherwise keep serving a stale master/flat playlist after the mode changes.
        """
        filename = "master.m3u8" if self._subtitle_language is not None else "live.m3u8"
        path = os.path.join(self.hls_dir, filename)
        return path if os.path.exists(path) else None

    def get_segment_path(self, filename: str) -> Optional[str]:
        """Get path to a specific segment or sub-playlist file."""
        # Sanitize filename to prevent directory traversal
        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith((".ts", ".vtt", ".m3u8")):
            return None

        path = os.path.join(self.hls_dir, safe_filename)
        return path if os.path.exists(path) else None


class BroadcastManager:
    """
    Manages multiple network broadcasts.

    Coordinates:
    - Starting/stopping broadcasts
    - Programme transitions with segment continuity
    - HLS directory lifecycle
    """

    def __init__(self, hls_base_dir: Optional[str] = None):
        self.hls_base_dir = hls_base_dir or getattr(
            settings, "HLS_BROADCAST_DIR", "/tmp/m3u-proxy-broadcasts"
        )
        self.dvr_base_dir: str = getattr(
            settings, "DVR_RECORDING_DIR", "/tmp/m3u-proxy-dvr"
        )
        self.broadcasts: Dict[str, NetworkBroadcastProcess] = {}
        self._lock = asyncio.Lock()

        # Start attempt tracking (avoid infinite restart loops)
        # Map network_id -> {count: int, first_attempt_at: float}
        self._start_attempts: Dict[str, dict] = {}

        # Configuration (can be overridden via settings)
        self.MAX_START_RETRIES = int(
            getattr(settings, "BROADCAST_MAX_START_RETRIES", 3)
        )
        self.START_RETRY_WINDOW = float(
            getattr(settings, "BROADCAST_START_RETRY_WINDOW", 300.0)
        )
        self.START_RETRY_COOLDOWN = float(
            getattr(settings, "BROADCAST_START_RETRY_COOLDOWN", 15.0)
        )
        self.START_FAILURE_GRACE = float(
            getattr(settings, "BROADCAST_START_FAILURE_GRACE", 2.0)
        )

        # Broadcast GC configuration (reuses HLS_GC_* thresholds)
        self.broadcast_gc_enabled = bool(
            getattr(settings, "BROADCAST_GC_ENABLED", True)
        )
        self.broadcast_gc_interval = int(getattr(settings, "HLS_GC_INTERVAL", 600))
        self.broadcast_gc_age_threshold = int(
            getattr(settings, "HLS_GC_AGE_THRESHOLD", 3600)
        )
        self._gc_task: Optional[asyncio.Task] = None

        # Ensure base directory exists
        os.makedirs(self.hls_base_dir, exist_ok=True)
        logger.info(f"BroadcastManager initialized with base dir: {self.hls_base_dir}")

    async def start_broadcast(self, config: BroadcastConfig) -> BroadcastStatus:
        """
        Start or transition a network broadcast.

        If a broadcast is already running for this network, it will be stopped
        gracefully and the new broadcast will continue with the next segment number.
        """
        async with self._lock:
            network_id = config.network_id

            # Check if broadcast already running
            if network_id in self.broadcasts:
                existing = self.broadcasts[network_id]
                logger.info(f"Transitioning broadcast {network_id} to new programme")

                # Stop existing process gracefully
                final_segment = await existing.stop(graceful=True)

                # Auto-continue segment numbering if not specified
                if config.segment_start_number == 0:
                    config.segment_start_number = final_segment + 1
                    # Force discontinuity on transition
                    config.add_discontinuity = True

                # Clean up segments no longer referenced by the playlist before handing
                # off to the new FFmpeg process. The playlist itself is left in place so
                # the new process can overwrite it with a discontinuity marker.
                existing.cleanup_orphaned_segments(age_threshold=0)

                del self.broadcasts[network_id]

            # Create and start new process
            # DVR recordings use a dedicated directory separate from live broadcasts
            base_dir = (
                getattr(settings, "DVR_RECORDING_DIR", "/tmp/m3u-proxy-dvr")
                if config.dvr_mode
                else self.hls_base_dir
            )
            process = NetworkBroadcastProcess(config, base_dir)

            # Check start retry policy
            now = time.time()
            attempts = self._start_attempts.get(network_id)
            if attempts:
                # reset window if expired
                if (
                    now - attempts.get("first_attempt_at", now)
                    > self.START_RETRY_WINDOW
                ):
                    attempts = None
                    del self._start_attempts[network_id]

            # If we've hit the max retries, check cooldown period to allow automatic retry
            if attempts and attempts.get("count", 0) >= self.MAX_START_RETRIES:
                last = attempts.get(
                    "last_attempt_at", attempts.get("first_attempt_at", now)
                )
                # If cooldown elapsed, clear attempts and allow retry
                if now - last >= self.START_RETRY_COOLDOWN:
                    logger.info(
                        f"Cooldown elapsed for broadcast {network_id}; resetting start retry counter and allowing automatic start."
                    )
                    del self._start_attempts[network_id]
                    attempts = None
                else:
                    seconds_left = int(self.START_RETRY_COOLDOWN - (now - last))
                    logger.error(
                        f"Exceeded max start retries ({self.MAX_START_RETRIES}) for broadcast {network_id}; refusing to start for another {seconds_left}s."
                    )
                    raise RuntimeError(
                        f"Exceeded max start retries for broadcast {network_id}; retry allowed after {seconds_left}s"
                    )

            success = await process.start()

            if not success:
                # Record failure immediately
                at = self._start_attempts.setdefault(
                    network_id,
                    {"count": 0, "first_attempt_at": now, "last_attempt_at": now},
                )
                at["count"] += 1
                at["last_attempt_at"] = now
                logger.warning(
                    f"Start attempt {at['count']} failed for {network_id}: {process.error_message}"
                )
                raise RuntimeError(
                    f"Failed to start broadcast: {process.error_message}"
                )

            # Add to active broadcasts and give a short grace period to detect immediate failures
            self.broadcasts[network_id] = process

            # Wait a small grace period to detect immediate startup failures (e.g., input errors)
            try:
                await asyncio.sleep(self.START_FAILURE_GRACE)
            except asyncio.CancelledError:
                pass

            # If process already failed within the grace period, treat as a start failure
            if process.status == "failed" or (
                process.process
                and process.process.returncode is not None
                and process.process.returncode != 0
            ):
                at = self._start_attempts.setdefault(
                    network_id,
                    {"count": 0, "first_attempt_at": now, "last_attempt_at": now},
                )
                at["count"] += 1
                at["last_attempt_at"] = now
                logger.warning(
                    f"Start attempt {at['count']} failed (post-start) for {network_id}: {process.error_message}"
                )

                # Clean up the failed process to avoid stale entries
                try:
                    await process.stop(graceful=False)
                except Exception:
                    pass
                if network_id in self.broadcasts:
                    del self.broadcasts[network_id]

                # If we've exceeded attempts, log an error (cooldown will be enforced on next start attempt)
                if at["count"] >= self.MAX_START_RETRIES:
                    logger.error(
                        f"Exceeded max start retries ({self.MAX_START_RETRIES}) for broadcast {network_id}; refusing further automatic starts until cooldown elapses."
                    )
                raise RuntimeError(
                    f"Broadcast {network_id} failed shortly after start: {process.error_message}"
                )

            # Successful start; clear any previous attempts
            if network_id in self._start_attempts:
                del self._start_attempts[network_id]

            return process.get_status()

    async def stop_broadcast(self, network_id: str) -> Optional[BroadcastStatus]:
        """Stop a network broadcast and clean up."""
        async with self._lock:
            if network_id not in self.broadcasts:
                return None

            process = self.broadcasts[network_id]
            await process.stop(graceful=True)

            status = process.get_status()
            del self.broadcasts[network_id]

            # Reset any tracked start attempts for this network since we stopped it manually
            if network_id in self._start_attempts:
                del self._start_attempts[network_id]

            return status

    def get_status(self, network_id: str) -> Optional[BroadcastStatus]:
        """Get current broadcast status."""
        if network_id not in self.broadcasts:
            return None
        return self.broadcasts[network_id].get_status()

    def get_all_statuses(self) -> Dict[str, BroadcastStatus]:
        """Get status of all active broadcasts."""
        return {
            network_id: process.get_status()
            for network_id, process in self.broadcasts.items()
        }

    async def read_playlist(self, network_id: str) -> Optional[str]:
        """Read the HLS playlist content for a network."""
        if network_id not in self.broadcasts:
            # Check if directory exists even without active broadcast (for recovery).
            # Active broadcasts with subtitles write master.m3u8; fall back to live.m3u8.
            # DVR recordings (post-broadcast) live in dvr_base_dir and only have live.m3u8.
            playlist_path = None
            hls_broadcast_dir = os.path.join(self.hls_base_dir, f"broadcast_{network_id}")
            for candidate_name in ("master.m3u8", "live.m3u8"):
                candidate = os.path.join(hls_broadcast_dir, candidate_name)
                if os.path.exists(candidate):
                    playlist_path = candidate
                    break
            if playlist_path is None:
                dvr_candidate = os.path.join(
                    self.dvr_base_dir, f"broadcast_{network_id}", "live.m3u8"
                )
                if os.path.exists(dvr_candidate):
                    playlist_path = dvr_candidate
            if playlist_path is None:
                return None
            if os.path.exists(playlist_path):
                try:
                    with open(playlist_path, "r") as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading playlist for {network_id}: {e}")
            return None

        process = self.broadcasts[network_id]
        playlist_path = process.get_playlist_path()

        if not playlist_path:
            return None

        try:
            with open(playlist_path, "r") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading playlist for {network_id}: {e}")
            return None

    def get_segment_path(self, network_id: str, filename: str) -> Optional[str]:
        """Get path to a segment or sub-playlist file for a network."""
        # Sanitize filename
        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith((".ts", ".vtt", ".m3u8")):
            return None

        # Check active broadcast first
        if network_id in self.broadcasts:
            return self.broadcasts[network_id].get_segment_path(filename)

        # Check directory even without active broadcast.
        # DVR recordings use dvr_base_dir; live broadcasts use hls_base_dir.
        for base_dir in (self.dvr_base_dir, self.hls_base_dir):
            segment_path = os.path.join(
                base_dir, f"broadcast_{network_id}", safe_filename
            )
            if os.path.exists(segment_path):
                return segment_path
        return None

    async def cleanup_broadcast(self, network_id: str) -> bool:
        """Clean up broadcast directory and files."""
        async with self._lock:
            # Stop if running
            if network_id in self.broadcasts:
                await self.broadcasts[network_id].stop(graceful=False)
                del self.broadcasts[network_id]

            # Remove directory
            broadcast_dir = os.path.join(self.hls_base_dir, f"broadcast_{network_id}")
            if os.path.exists(broadcast_dir):
                try:
                    import shutil

                    shutil.rmtree(broadcast_dir)
                    logger.info(f"Cleaned up broadcast directory: {broadcast_dir}")
                    # Clear start attempts on successful cleanup
                    if network_id in self._start_attempts:
                        del self._start_attempts[network_id]
                    return True
                except Exception as e:
                    logger.error(f"Error cleaning up broadcast {network_id}: {e}")
                    return False

            return True

    async def start(self):
        """Start background tasks (GC loop)."""
        if self.broadcast_gc_enabled:
            self._gc_task = asyncio.create_task(self._gc_loop())
            logger.info(
                f"Broadcast GC started (interval={self.broadcast_gc_interval}s, "
                f"age_threshold={self.broadcast_gc_age_threshold}s)"
            )

    async def _gc_loop(self):
        """Periodically scan for and remove stale broadcast directories."""
        while True:
            try:
                await asyncio.sleep(self.broadcast_gc_interval)
                await self._gc_broadcast_dirs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast GC loop error: {e}")

    async def _gc_broadcast_dirs(self):
        """
        Two-phase broadcast GC:

        1. Active broadcasts — remove .ts files that are no longer referenced by the
           current playlist and are older than 60 seconds (guard against race with
           a concurrent transition writing new segments).
        2. Inactive directories — remove entire stale broadcast dirs (no active process,
           age > gc_age_threshold) using shutil.rmtree.
        """
        dirs_removed = skipped_too_young = 0

        # Snapshot active processes without holding the lock during I/O
        async with self._lock:
            active_snapshot = {
                hls_dir: process
                for hls_dir, process in (
                    (p.hls_dir, p) for p in self.broadcasts.values()
                )
            }

        # Phase 1: clean orphaned segments from active broadcasts
        for process in active_snapshot.values():
            process.cleanup_orphaned_segments(age_threshold=60)

        # Phase 2: remove entire stale inactive directories
        try:
            entries = os.listdir(self.hls_base_dir)
        except Exception as e:
            logger.error(f"Broadcast GC: cannot list {self.hls_base_dir}: {e}")
            return

        now = time.time()
        for entry in entries:
            if not entry.startswith("broadcast_"):
                continue

            full_path = os.path.join(self.hls_base_dir, entry)
            if not os.path.isdir(full_path):
                continue

            if full_path in active_snapshot:
                continue

            try:
                age = now - os.path.getmtime(full_path)
            except Exception:
                continue

            if age < self.broadcast_gc_age_threshold:
                skipped_too_young += 1
                continue

            try:
                shutil.rmtree(full_path)
                dirs_removed += 1
                logger.info(
                    f"Broadcast GC: removed stale directory {full_path} (age={age:.0f}s)"
                )
            except Exception as e:
                logger.error(f"Broadcast GC: failed to remove {full_path}: {e}")

        if dirs_removed or skipped_too_young:
            logger.info(
                f"Broadcast GC: dirs_removed={dirs_removed}, skipped_too_young={skipped_too_young}"
            )

    async def shutdown(self):
        """Stop all broadcasts gracefully."""
        logger.info("Shutting down BroadcastManager...")

        if self._gc_task and not self._gc_task.done():
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for network_id, process in list(self.broadcasts.items()):
                try:
                    await process.stop(graceful=True)
                except Exception as e:
                    logger.error(f"Error stopping broadcast {network_id}: {e}")

            self.broadcasts.clear()
        logger.info("BroadcastManager shutdown complete")


# Global instance (initialized in api.py lifespan)
broadcast_manager: Optional[BroadcastManager] = None
