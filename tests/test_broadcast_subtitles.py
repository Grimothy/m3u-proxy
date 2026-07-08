import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.broadcast_manager import BroadcastConfig, NetworkBroadcastProcess


def _proc_with_language(language, **config_kwargs):
    cfg = BroadcastConfig(
        network_id="subtest",
        stream_url="http://example.com/video.mkv",
        subtitles_enabled=True,
        **config_kwargs,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = language
    return proc


def _mock_httpx_get(status_code=200, content=b"1\n00:00:01,000 --> 00:00:02,000\nHi\n"):
    """Mock httpx.AsyncClient()... .get(url) for _subtitle_url_has_content()."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.content = content

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return patch("httpx.AsyncClient", return_value=mock_client)


def test_command_unchanged_when_subtitles_disabled():
    cfg = BroadcastConfig(
        network_id="nosub", stream_url="http://example.com/video.mkv"
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    cmd = proc._build_ffmpeg_command()

    assert "-var_stream_map" not in cmd
    assert "-c:s" not in cmd
    assert cmd[-1] == "/tmp/broadcast_nosub/live.m3u8"
    assert "-hls_segment_filename" in cmd
    seg_idx = cmd.index("-hls_segment_filename")
    assert cmd[seg_idx + 1] == "/tmp/broadcast_nosub/live%06d.ts"


def test_command_falls_back_when_subtitles_enabled_but_none_detected():
    proc = _proc_with_language(None)
    cmd = proc._build_ffmpeg_command()

    # No subtitle stream was found by the probe -> identical to the disabled case.
    assert "-var_stream_map" not in cmd
    assert "-c:s" not in cmd
    assert cmd[-1] == "/tmp/broadcast_subtest/live.m3u8"


def test_command_builds_master_variant_when_subtitle_detected():
    proc = _proc_with_language("eng")
    cmd = proc._build_ffmpeg_command()

    assert "-map" in cmd
    assert "0:s:0?" in cmd
    assert "-c:s" in cmd
    assert cmd[cmd.index("-c:s") + 1] == "webvtt"

    assert "-var_stream_map" in cmd
    var_stream_map = cmd[cmd.index("-var_stream_map") + 1]
    assert var_stream_map == "v:0,a:0,s:0,sgroup:subs,language:eng"

    assert "-master_pl_name" in cmd
    assert cmd[cmd.index("-master_pl_name") + 1] == "master.m3u8"

    seg_idx = cmd.index("-hls_segment_filename")
    assert cmd[seg_idx + 1] == "/tmp/broadcast_subtest/live%v_%06d.ts"
    assert cmd[-1] == "/tmp/broadcast_subtest/live_%v.m3u8"


def test_command_omits_language_attribute_when_subtitle_untagged():
    proc = _proc_with_language("")
    cmd = proc._build_ffmpeg_command()

    var_stream_map = cmd[cmd.index("-var_stream_map") + 1]
    assert var_stream_map == "v:0,a:0,s:0,sgroup:subs"
    assert "language:" not in var_stream_map


def test_command_adds_manifest_only_bitrate_hints_in_copy_mode():
    proc = _proc_with_language("eng", transcode=False)
    cmd = proc._build_ffmpeg_command()

    # -c:v/-c:a copy must be unaffected (no re-encode) even with hints added.
    assert "-c:v" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert "-b:v" in cmd
    assert "-b:a" in cmd


def test_command_does_not_duplicate_bitrate_flags_in_transcode_mode():
    proc = _proc_with_language(
        "eng", transcode=True, video_bitrate="4000", audio_bitrate=256
    )
    cmd = proc._build_ffmpeg_command()

    assert cmd.count("-b:v") == 1
    assert cmd.count("-b:a") == 1
    assert cmd[cmd.index("-b:v") + 1] == "4000k"
    assert cmd[cmd.index("-b:a") + 1] == "256k"


@pytest.mark.asyncio
async def test_probe_subtitle_language_returns_language_from_ffprobe():
    cfg = BroadcastConfig(
        network_id="probetest",
        stream_url="http://example.com/video.mkv",
        subtitles_enabled=True,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    fake_stdout = json.dumps(
        {
            "streams": [
                {"index": 2, "codec_name": "subrip", "tags": {"language": "eng"}}
            ]
        }
    ).encode()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(fake_stdout, b""))
    mock_proc.returncode = 0

    with patch(
        "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)
    ):
        language = await proc._probe_subtitle_language()

    assert language == "eng"


@pytest.mark.asyncio
async def test_probe_subtitle_language_returns_none_for_bitmap_subtitle_codec():
    """
    Bitmap subtitle formats (PGS, VobSub/dvd_subtitle, DVB) can't be transcoded
    to WebVTT — FFmpeg aborts the entire process (video+audio included) if asked
    to. The probe must reject these so the broadcast falls back to no subtitles
    instead of crash-looping.
    """
    cfg = BroadcastConfig(
        network_id="probetest-bitmap",
        stream_url="http://example.com/video.mkv",
        subtitles_enabled=True,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    fake_stdout = json.dumps(
        {
            "streams": [
                {
                    "index": 2,
                    "codec_name": "hdmv_pgs_subtitle",
                    "tags": {"language": "eng"},
                }
            ]
        }
    ).encode()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(fake_stdout, b""))
    mock_proc.returncode = 0

    with patch(
        "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)
    ):
        language = await proc._probe_subtitle_language()

    assert language is None


@pytest.mark.asyncio
async def test_probe_subtitle_language_returns_none_when_no_streams():
    cfg = BroadcastConfig(
        network_id="probetest2",
        stream_url="http://example.com/video.mkv",
        subtitles_enabled=True,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b'{"streams": []}', b""))
    mock_proc.returncode = 0

    with patch(
        "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)
    ):
        language = await proc._probe_subtitle_language()

    assert language is None


@pytest.mark.asyncio
async def test_probe_subtitle_language_fails_closed_on_probe_error():
    cfg = BroadcastConfig(
        network_id="probetest3",
        stream_url="http://example.com/video.mkv",
        subtitles_enabled=True,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
    mock_proc.returncode = 1

    with patch(
        "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)
    ):
        language = await proc._probe_subtitle_language()

    assert language is None


def test_final_segment_number_matches_variant_prefixed_filenames(tmp_path):
    hls_dir = tmp_path / "broadcast_segnum"
    hls_dir.mkdir()
    (hls_dir / "live0_000000.ts").write_bytes(b"x")
    (hls_dir / "live0_000005.ts").write_bytes(b"x")

    cfg = BroadcastConfig(network_id="segnum", stream_url="http://example.com/v.mkv")
    proc = NetworkBroadcastProcess(cfg, hls_base_dir=str(tmp_path))

    assert proc._get_final_segment_number() == 5


def test_get_playlist_path_uses_current_subtitle_state_not_disk_presence(tmp_path):
    """
    Regression test: get_playlist_path() must be keyed off self._subtitle_language
    (the CURRENT ffmpeg invocation's state), not "does master.m3u8 exist on disk".
    A stale master.m3u8 left over from a previous, subtitled programme (stale-file
    cleanup is skipped during transitions to preserve segment continuity) must NOT
    be served once the current programme has no subtitles.
    """
    hls_dir = tmp_path / "broadcast_pathtest"
    hls_dir.mkdir()
    (hls_dir / "live.m3u8").write_text("flat")
    (hls_dir / "master.m3u8").write_text("stale master from a previous subtitled programme")

    cfg = BroadcastConfig(network_id="pathtest", stream_url="http://example.com/v.mkv")
    proc = NetworkBroadcastProcess(cfg, hls_base_dir=str(tmp_path))

    proc._subtitle_language = None
    assert proc.get_playlist_path() == str(hls_dir / "live.m3u8")

    proc._subtitle_language = "eng"
    assert proc.get_playlist_path() == str(hls_dir / "master.m3u8")


def test_get_segment_path_allows_vtt_and_m3u8(tmp_path):
    hls_dir = tmp_path / "broadcast_segpath"
    hls_dir.mkdir()
    (hls_dir / "live_00.vtt").write_text("WEBVTT")
    (hls_dir / "live_0.m3u8").write_text("#EXTM3U")
    (hls_dir / "notes.txt").write_text("nope")

    cfg = BroadcastConfig(network_id="segpath", stream_url="http://example.com/v.mkv")
    proc = NetworkBroadcastProcess(cfg, hls_base_dir=str(tmp_path))

    assert proc.get_segment_path("live_00.vtt") == str(hls_dir / "live_00.vtt")
    assert proc.get_segment_path("live_0.m3u8") == str(hls_dir / "live_0.m3u8")
    assert proc.get_segment_path("notes.txt") is None


def test_cleanup_orphaned_segments_handles_master_variant_layout(tmp_path):
    hls_dir = tmp_path / "broadcast_cleanup"
    hls_dir.mkdir()
    (hls_dir / "master.m3u8").write_text("#EXTM3U\n")
    (hls_dir / "live_0.m3u8").write_text("#EXTM3U\nlive0_000001.ts\n")
    (hls_dir / "live_0_vtt.m3u8").write_text("#EXTM3U\nlive_001.vtt\n")
    (hls_dir / "live0_000001.ts").write_bytes(b"x")  # referenced -> keep
    (hls_dir / "live0_000000.ts").write_bytes(b"x")  # orphaned -> remove
    (hls_dir / "live_001.vtt").write_text("WEBVTT")  # referenced -> keep
    (hls_dir / "live_000.vtt").write_text("WEBVTT")  # orphaned -> remove

    cfg = BroadcastConfig(network_id="cleanup", stream_url="http://example.com/v.mkv")
    proc = NetworkBroadcastProcess(cfg, hls_base_dir=str(tmp_path))
    proc._subtitle_language = "eng"  # current invocation has subtitles active

    removed = proc.cleanup_orphaned_segments(age_threshold=0)

    assert removed == 2
    assert (hls_dir / "live0_000001.ts").exists()
    assert (hls_dir / "live_001.vtt").exists()
    assert not (hls_dir / "live0_000000.ts").exists()
    assert not (hls_dir / "live_000.vtt").exists()
    # Manifest files are never treated as orphaned segments.
    assert (hls_dir / "master.m3u8").exists()
    assert (hls_dir / "live_0.m3u8").exists()
    assert (hls_dir / "live_0_vtt.m3u8").exists()


def test_cleanup_orphaned_segments_ignores_stale_master_after_transition_to_flat(
    tmp_path,
):
    """
    Regression test for a transition bug: after a program transition from
    subtitled -> non-subtitled content, a stale master.m3u8/live_0*.m3u8 from the
    previous invocation can still be on disk (stale-file cleanup is skipped during
    transitions). cleanup_orphaned_segments() must use the flat live.m3u8 (the
    CURRENT invocation's manifest), not fall back to the stale master layout just
    because master.m3u8 happens to still exist.
    """
    hls_dir = tmp_path / "broadcast_stalemaster"
    hls_dir.mkdir()
    # Stale leftovers from the previous (subtitled) programme.
    (hls_dir / "master.m3u8").write_text("#EXTM3U\n")
    (hls_dir / "live_0.m3u8").write_text("#EXTM3U\nlive0_000000.ts\n")
    (hls_dir / "live_0_vtt.m3u8").write_text("#EXTM3U\nlive_00.vtt\n")
    # Current (non-subtitled) invocation's real manifest and segment.
    (hls_dir / "live.m3u8").write_text("#EXTM3U\nlive000001.ts\n")
    (hls_dir / "live000001.ts").write_bytes(b"x")  # referenced by live.m3u8 -> keep

    cfg = BroadcastConfig(network_id="stalemaster", stream_url="http://example.com/v.mkv")
    proc = NetworkBroadcastProcess(cfg, hls_base_dir=str(tmp_path))
    proc._subtitle_language = None  # current invocation has no subtitles

    removed = proc.cleanup_orphaned_segments(age_threshold=0)

    assert removed == 0
    assert (hls_dir / "live000001.ts").exists()


def test_command_adds_second_input_for_explicit_subtitle_url():
    cfg = BroadcastConfig(
        network_id="exturl",
        stream_url="http://emby.local/video.ts",
        subtitle_url="http://emby.local/Videos/1/mediasource_1/Subtitles/2/Stream.srt",
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = "eng"
    proc._subtitle_input_index = 1

    cmd = proc._build_ffmpeg_command()

    # Both inputs present, subtitle input after the main video input.
    i_indices = [i for i, arg in enumerate(cmd) if arg == "-i"]
    assert len(i_indices) == 2
    assert cmd[i_indices[0] + 1] == "http://emby.local/video.ts"
    assert cmd[i_indices[1] + 1] == cfg.subtitle_url

    # Subtitle mapped from input 1, not input 0.
    assert "1:s:0?" in cmd
    assert "0:s:0?" not in cmd

    # -t must come after BOTH -i's, or ffmpeg would scope it to the subtitle input
    # instead of applying it as the output-level duration limit.
    t_idx = cmd.index("-t") if "-t" in cmd else None
    if t_idx is not None:
        assert t_idx > i_indices[1]


def test_passing_both_seek_seconds_zero_and_subtitle_seek_seconds_zero_applies_neither():
    """
    CONTRACT test: the ONLY way to signal "no seek needed" is to send both
    seek_seconds=0 AND subtitle_seek_seconds=0. When that combination arrives, the
    proxy must NOT apply -ss to the video input AND must NOT apply -ss/-itsoffset
    to the subtitle input. This locks the single-authority contract: the media server
    is the sole seek authority for both streams, and the proxy leaves them untouched.
    """
    cfg = BroadcastConfig(
        network_id="preseek",
        # Video seeked server-side (StartTimeTicks) -> seek_seconds zeroed by Laravel.
        stream_url="http://emby.local/video.ts?StartTimeTicks=84500000000&VideoCodec=copy",
        # Subtitle URL carries the startPositionTicks path segment: already rebased.
        subtitle_url="http://emby.local/Videos/1/mediasource_1/Subtitles/3/8450000000/Stream.srt",
        seek_seconds=0,
        subtitle_seek_seconds=0,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = "eng"
    proc._subtitle_input_index = 1

    cmd = proc._build_ffmpeg_command()

    # The pre-seeked subtitle URL is added as-is, with no local seeking of any input.
    i_indices = [i for i, arg in enumerate(cmd) if arg == "-i"]
    assert len(i_indices) == 2
    assert cmd[i_indices[1] + 1] == cfg.subtitle_url
    assert "-ss" not in cmd
    assert "-itsoffset" not in cmd


def test_command_applies_input_seek_to_rewritten_static_url():
    """
    Regression test for the Emby VideoCodec=copy remux seek bug: PHP rewrites the
    remux URL (strips VideoCodec=copy AND StartTimeTicks, adds static=true) so the
    static endpoint is hit. Emby ignores StartTimeTicks server-side on BOTH static
    and remux (verified against a live Emby via md5). The static endpoint DOES
    support byte-range requests (Accept-Ranges: bytes), so the proxy must apply
    ffmpeg input -ss against it — that's how the seek actually happens.
    """
    cfg = BroadcastConfig(
        network_id="remux_rewritten",
        # What PHP sends after rewrite: static URL with AudioStreamIndex.
        # VideoCodec=copy and StartTimeTicks were both stripped (Emby ignores
        # them on static), static=true was added.
        stream_url="http://emby.local/Videos/1/stream.ts?api_key=abc&AudioStreamIndex=1&static=true",
        seek_seconds=2715,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    cmd = proc._build_ffmpeg_command()

    # Input-level -ss must be present, before the main -i, with the correct value.
    assert "-ss" in cmd
    ss_idx = cmd.index("-ss")
    assert cmd[ss_idx + 1] == "2715"

    i_indices = [i for i, arg in enumerate(cmd) if arg == "-i"]
    assert ss_idx < i_indices[0]


def test_command_applies_input_seek_when_stream_url_has_seek_param():
    """
    Regression test for the Emby VideoCodec=copy remux seek bug: PHP rewrites the
    remux URL to the seek-capable static endpoint (strips VideoCodec=copy, keeps
    StartTimeTicks + AudioStreamIndex) and sends seek_seconds > 0. The proxy must
    apply ffmpeg input -ss so the seek actually happens. Before this fix PHP
    zeroed seek_seconds for any remux URL, so ffmpeg played from byte 0.
    """
    cfg = BroadcastConfig(
        network_id="remux_seek",
        # PHP-rewritten URL: VideoCodec=copy stripped, StartTimeTicks kept.
        # Looks like a static+seeked URL because that's what it now is.
        stream_url="http://emby.local/Videos/1/stream.ts?api_key=abc&StartTimeTicks=27150000000&AudioStreamIndex=1",
        seek_seconds=2715,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = "eng"
    proc._subtitle_input_index = 1

    cmd = proc._build_ffmpeg_command()

    # Input-level -ss must be present, before the main -i, with the correct value.
    assert "-ss" in cmd
    ss_idx = cmd.index("-ss")
    assert cmd[ss_idx + 1] == "2715"

    i_indices = [i for i, arg in enumerate(cmd) if arg == "-i"]
    assert ss_idx < i_indices[0]

    # The full rebase-to-zero anti-desync contract still holds: the subtitle input
    # arrives pre-seeked by Emby (subtitle_seek_seconds=0), so NO -ss/-itsoffset on it.
    assert "-itsoffset" not in cmd


def test_command_seeks_subtitle_input_to_match_video_seek():
    cfg = BroadcastConfig(
        network_id="seeksub",
        stream_url="http://emby.local/video.ts",
        subtitle_url="http://emby.local/sub.srt",
        seek_seconds=120,
        # Explicit fallback-path setup: PHP sends the same seek value for the
        # subtitle input. (When subtitle_seek_seconds is omitted the proxy now
        # falls back to 0 — see test_subtitle_input_falls_back_to_zero_when_*
        # — so this test must set it explicitly to exercise the
        # "-ss + -itsoffset" branch.)
        subtitle_seek_seconds=120,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = "eng"
    proc._subtitle_input_index = 1

    cmd = proc._build_ffmpeg_command()

    i_indices = [i for i, arg in enumerate(cmd) if arg == "-i"]
    ss_indices = [i for i, arg in enumerate(cmd) if arg == "-ss"]

    # One -ss before each -i (video input, then subtitle input).
    assert len(ss_indices) == 2
    for ss_idx in ss_indices:
        assert cmd[ss_idx + 1] == "120"
    assert ss_indices[0] < i_indices[0]
    assert ss_indices[1] < i_indices[1]

    # The subtitle input also gets a negative -itsoffset to compensate for the
    # seek rebasing its output timestamps back to absolute (see
    # test_command_shifts_subtitle_timestamps_to_match_rebased_video_pts).
    assert "-itsoffset" in cmd
    itsoffset_idx = cmd.index("-itsoffset")
    assert cmd[itsoffset_idx + 1] == "-120"


def test_subtitle_input_falls_back_to_zero_when_subtitle_seek_seconds_omitted():
    """
    When subtitle_seek_seconds is not set (None), the proxy must NOT fall back to
    seek_seconds for the subtitle input — it must fall back to 0 (no seek).
    Fallback to seek_seconds would apply a -ss/-itsoffset offset meant for the VIDEO
    input to the SUBTITLE input, pushing subtitle cue timestamps out of sync.
    """
    cfg = BroadcastConfig(
        network_id="fallback-zero",
        stream_url="http://emby.local/video.ts",
        subtitle_url="http://emby.local/sub.srt",
        seek_seconds=2715,
        # subtitle_seek_seconds intentionally omitted (None)
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = "eng"
    proc._subtitle_input_index = 1

    cmd = proc._build_ffmpeg_command()

    # Video input gets -ss 2715 as expected.
    assert "-ss" in cmd
    ss_idx = cmd.index("-ss")
    assert cmd[ss_idx + 1] == "2715"

    # Subtitle input gets NO -ss / -itsoffset — fallback is 0, not seek_seconds.
    i_indices = [i for i, arg in enumerate(cmd) if arg == "-i"]
    assert len(i_indices) == 2
    # -ss appears before the FIRST -i (video input), nothing before the second.
    assert ss_idx < i_indices[0]
    assert "-itsoffset" not in cmd


def test_command_shifts_subtitle_timestamps_to_match_rebased_video_pts():
    """
    Regression test for a real production desync: subtitles playing badly out of
    sync (dialogue that looks like it's "for a different movie") whenever a
    broadcast resumes with a non-zero seek.

    Root cause, confirmed against a live Emby source with ffprobe: a
    VideoCodec=copy remux REBASES the video's own PTS to ~0 at the seek point
    (first video packet PTS ≈ 0, not ≈ the seek offset). But ffmpeg's -ss on the
    subtitle_url input only skips early cues — it does NOT rebase the surviving
    ones, which stay on the subtitle file's own absolute timeline. Without
    correcting for that mismatch, every subtitle cue ends up subtitle_seek
    seconds later than the video position it's meant to caption. -itsoffset
    (negative, equal in magnitude to the seek) shifts the subtitle input back
    onto the video's rebased timeline.
    """
    cfg = BroadcastConfig(
        network_id="rebase",
        stream_url="http://emby.local/video.ts",
        subtitle_url="http://emby.local/sub.srt",
        subtitle_seek_seconds=69,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = "eng"
    proc._subtitle_input_index = 1

    cmd = proc._build_ffmpeg_command()

    i_indices = [i for i, arg in enumerate(cmd) if arg == "-i"]
    ss_idx = cmd.index("-ss")
    itsoffset_idx = cmd.index("-itsoffset")

    assert cmd[ss_idx + 1] == "69"
    assert cmd[itsoffset_idx + 1] == "-69"
    # Both must land before the subtitle -i, and -itsoffset must not leak onto
    # the main video input.
    assert ss_idx < i_indices[1]
    assert itsoffset_idx < i_indices[1]
    assert cmd[: i_indices[0] + 2].count("-itsoffset") == 0


def test_command_omits_itsoffset_when_there_is_no_seek():
    cfg = BroadcastConfig(
        network_id="noseek",
        stream_url="http://emby.local/video.ts",
        subtitle_url="http://emby.local/sub.srt",
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = "eng"
    proc._subtitle_input_index = 1

    cmd = proc._build_ffmpeg_command()

    assert "-itsoffset" not in cmd


def test_command_reconnects_the_subtitle_input_on_dropped_connections():
    cfg = BroadcastConfig(
        network_id="subreconnect",
        stream_url="http://emby.local/video.ts",
        subtitle_url="http://emby.local/sub.srt",
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = "eng"
    proc._subtitle_input_index = 1

    cmd = proc._build_ffmpeg_command()

    i_indices = [i for i, arg in enumerate(cmd) if arg == "-i"]
    between_inputs = cmd[i_indices[0] + 1 : i_indices[1]]

    # The subtitle input must get its own reconnect options — without them,
    # Emby/Jellyfin closing this connection mid-broadcast leaves ffmpeg with a
    # dead subtitle input it never retries, silently dropping all further
    # subtitle cues for the rest of the broadcast.
    assert between_inputs.count("-reconnect") == 1
    assert between_inputs[between_inputs.index("-reconnect") + 1] == "1"
    assert between_inputs[between_inputs.index("-reconnect_streamed") + 1] == "1"
    assert between_inputs[between_inputs.index("-reconnect_delay_max") + 1] == "10"


@pytest.mark.asyncio
async def test_resolve_subtitle_state_prefers_explicit_url_and_skips_probing():
    cfg = BroadcastConfig(
        network_id="prefer-url",
        stream_url="http://emby.local/video.ts",
        subtitles_enabled=True,
        subtitle_url="http://emby.local/sub.srt",
        subtitle_language="fre",
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    with patch("asyncio.create_subprocess_exec") as mock_exec, _mock_httpx_get():
        await proc._resolve_subtitle_state()

    mock_exec.assert_not_called()
    assert proc._subtitle_language == "fre"
    assert proc._subtitle_input_index == 1


@pytest.mark.asyncio
async def test_resolve_subtitle_state_falls_back_when_subtitle_url_is_empty():
    """
    Media servers' server-side seeked subtitle endpoints (e.g. Emby/Jellyfin's
    startPositionTicks) can return HTTP 200 with a completely empty body when the
    seek lands past the subtitle track's last cue. FFmpeg can't open an empty file
    at all, which would otherwise abort the whole broadcast (video+audio included).
    """
    cfg = BroadcastConfig(
        network_id="empty-url",
        stream_url="http://emby.local/video.ts",
        subtitles_enabled=True,
        subtitle_url="http://emby.local/sub.srt",
        subtitle_language="eng",
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    with _mock_httpx_get(status_code=200, content=b""):
        await proc._resolve_subtitle_state()

    assert proc._subtitle_language is None
    assert proc._subtitle_input_index == 0


@pytest.mark.asyncio
async def test_resolve_subtitle_state_falls_back_when_subtitle_url_fetch_fails():
    cfg = BroadcastConfig(
        network_id="broken-url",
        stream_url="http://emby.local/video.ts",
        subtitles_enabled=True,
        subtitle_url="http://emby.local/sub.srt",
        subtitle_language="eng",
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    with patch("httpx.AsyncClient", side_effect=RuntimeError("connection failed")):
        await proc._resolve_subtitle_state()

    assert proc._subtitle_language is None
    assert proc._subtitle_input_index == 0


@pytest.mark.asyncio
async def test_resolve_subtitle_state_falls_back_to_probe_without_url():
    cfg = BroadcastConfig(
        network_id="fallback-probe",
        stream_url="http://emby.local/video.ts",
        subtitles_enabled=True,
        subtitle_url=None,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(
        return_value=(
            json.dumps(
                {"streams": [{"codec_name": "subrip", "tags": {"language": "spa"}}]}
            ).encode(),
            b"",
        )
    )
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        await proc._resolve_subtitle_state()

    assert proc._subtitle_language == "spa"
    assert proc._subtitle_input_index == 0


@pytest.mark.asyncio
async def test_resolve_subtitle_state_untagged_language_defaults_to_empty_string():
    cfg = BroadcastConfig(
        network_id="untagged-url",
        stream_url="http://emby.local/video.ts",
        subtitles_enabled=True,
        subtitle_url="http://emby.local/sub.srt",
        subtitle_language=None,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    with _mock_httpx_get():
        await proc._resolve_subtitle_state()

    assert proc._subtitle_language == ""
    assert proc._subtitle_input_index == 1


def test_command_subtitle_drift_correction_probe_applies_itsoffset(monkeypatch):
    """
    Regression test for the user-reported "subtitles a tad bit slow or fast" symptom
    in path F1: after a #range= byte-seek the video's first PTS lands at the GOP
    boundary (typically 1-3s offset). Subtitle cues land at exact PTS 0 because Emby's
    startPositionTicks rebases them. Apply +<first_video_pts> as -itsoffset on the
    subtitle input to align them.

    The probe is mocked here (real ffprobe would hit the network); this test exercises
    the command-builder logic, not the probe itself.
    """
    cfg = BroadcastConfig(
        network_id="drift_correction",
        # PHP-rewritten URL with #range= byte-seek.
        stream_url="http://emby.local/Videos/437/stream.ts?static=true#range=3556648482-",
        seek_seconds=4444,
        subtitle_url="http://emby.local/Videos/437/Subtitles/1/Stream.srt",
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = "eng"
    proc._subtitle_input_index = 1

    # Mock the probe to return 1.483 (the empirically measured GOP drift for item 437).
    # In a real broadcast this would be ffprobe hitting the network; we stub it here
    # to isolate the command-builder logic.
    monkeypatch.setattr(proc, "_probe_video_first_pts", lambda: 1.483)

    cmd = proc._build_ffmpeg_command()

    # The drift-correction -itsoffset +1.483 must be present on the subtitle input.
    itsoffset_indices = [i for i, arg in enumerate(cmd) if arg == "-itsoffset"]
    assert itsoffset_indices, "Expected at least one -itsoffset in command"
    assert any(cmd[i + 1] == "1.483" for i in itsoffset_indices), (
        f"Expected -itsoffset 1.483 somewhere in command; got: {cmd}"
    )


def test_command_no_drift_correction_when_probe_returns_zero(monkeypatch):
    """
    When the probe returns 0 (no meaningful drift, e.g. seek_seconds=0 or
    server-transcode mode where the server does the seeking), no -itsoffset is
    added beyond any pre-existing subtitle seek.
    """
    cfg = BroadcastConfig(
        network_id="no_drift",
        stream_url="http://emby.local/Videos/437/stream.ts?static=true",
        seek_seconds=0,
        subtitle_url="http://emby.local/Videos/437/Subtitles/1/Stream.srt",
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    proc._subtitle_language = "eng"
    proc._subtitle_input_index = 1
    monkeypatch.setattr(proc, "_probe_video_first_pts", lambda: 0.0)

    cmd = proc._build_ffmpeg_command()

    # No -itsoffset should be added when probe returns 0.
    assert "-itsoffset" not in cmd


def test_probe_video_first_pts_returns_correct_value(monkeypatch):
    """
    Verify _probe_video_first_pts() returns the value that ffprobe extracted from the
    tiny TS chunk that ffmpeg first wrote to disk. Both subprocess.run calls are mocked:
    the first writes a chunk to a temp file (succeeds silently), the second extracts
    PTS via ffprobe and returns the value.
    """
    cfg = BroadcastConfig(
        network_id="probe_value",
        stream_url="http://emby.local/Videos/437/stream.ts#range=3556648482-",
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")

    # ffprobe (the second subprocess.run call) returns PTS in stdout.
    # ffmpeg (the first call) succeeds silently with empty stdout.
    def fake_run(cmd, *args, **kwargs):
        # Heuristic: ffprobe is the one with `packet=pts_time` anywhere in args.
        cmd_str = " ".join(str(arg) for arg in cmd)
        if "packet=pts_time" in cmd_str:
            return MagicMock(returncode=0, stdout="1.483000\n")
        return MagicMock(returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    pts = proc._probe_video_first_pts()

    assert pts == 1.483
    # Second call should return the cached value (no second subprocess call).
    pts2 = proc._probe_video_first_pts()
    assert pts2 == 1.483
