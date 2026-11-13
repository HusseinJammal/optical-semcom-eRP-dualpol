# Transmitter DSP

| File | Function | What it does |
|---|---|---|
| `pilots.py` | `pilot_symbols`, `insert_pilots`, `remove_pilots`, `pilot_positions`, `pilot_amplitude` | fixed QPSK pilot on the 64-QAM corners (±1.080) before every 32 data symbols + a closing pilot: 256 → 265 symbols per polarization; different known sequences on X and Y |
| `framing.py` | `neighbour_stream`, `zero_padded_stream` | embeds the frame in random neighbour frames (64-QAM data at the frame's power + pilots on the same cadence) filling the TX pad and the guard on both sides |
| `pulse_shaping.py` | `upsample`, `rrc_filter`, `apply_filter`, `pad_guard`, ... | ×16 upsampling and a 257-tap, exactly symmetric, unit-energy RRC (roll-off 0.1); zero delay through the TX/RX cascade |
| `launch_power.py` | `set_frame_launch_power` | scales each image so the TOTAL power over both polarizations, measured over the frame, equals P |

The order of these blocks (and of the TX impairments between them) is fixed in `semcom/system/link.py`.
