from src.broadcast_manager import BroadcastConfig, NetworkBroadcastProcess


def test_command_defaults_to_first_audio_stream_when_no_index_given():
    cfg = BroadcastConfig(network_id="audiodefault", stream_url="http://example.com/video.mkv")
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    cmd = proc._build_ffmpeg_command()

    assert "0:a:0?" in cmd


def test_command_maps_explicit_audio_stream_index():
    """
    Laravel resolves audio_stream_index from the media server's MediaStreams
    metadata, which is an ABSOLUTE index spanning video+audio+subtitle streams
    together — not FFmpeg's type-relative "a:N" (Nth audio stream). Mapping it
    with a type letter silently maps nothing whenever the absolute index doesn't
    also happen to be a valid per-type audio position, which then aborts the
    whole HLS conversion when -var_stream_map's "a:0" reference goes dangling.
    """
    cfg = BroadcastConfig(
        network_id="audiotest",
        stream_url="http://example.com/video.mkv",
        audio_stream_index=2,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    cmd = proc._build_ffmpeg_command()

    assert "0:2?" in cmd
    assert "0:a:2?" not in cmd
    assert "0:a:0?" not in cmd


def test_command_maps_audio_stream_index_zero_explicitly():
    """
    Regression guard: audio_stream_index=0 must not be treated as falsy/unset —
    only None means "no preference". An explicit 0 still uses the absolute-index
    form ("0:0?"), distinct from the type-relative default ("0:a:0?").
    """
    cfg = BroadcastConfig(
        network_id="audiozero",
        stream_url="http://example.com/video.mkv",
        audio_stream_index=0,
    )
    proc = NetworkBroadcastProcess(cfg, hls_base_dir="/tmp")
    cmd = proc._build_ffmpeg_command()

    assert "0:0?" in cmd
    assert "0:a:0?" not in cmd
    map_indices = [i for i, arg in enumerate(cmd) if arg == "-map"]
    audio_maps = [cmd[i + 1] for i in map_indices if cmd[i + 1].startswith("0:") and not cmd[i + 1].startswith("0:v:")]
    assert audio_maps == ["0:0?"]
