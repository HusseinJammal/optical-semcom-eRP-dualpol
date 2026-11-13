# Receiver DSP

| File | Function | What it does |
|---|---|---|
| `cd_compensation.py` | `cd_compensate` | static, ideal frequency-domain CD compensation over the propagated length (phase only) |
| `sampling.py` | `matched_filter`, `strip_guard`, `downsample`, `extract_frame` | the TX RRC as matched filter, guard removal, 1 sample per symbol at the ideal instant (zero cascade delay), frame extraction |
| `cpr.py` | `pilot_cpr`, `genie_cpr` | pilot-aided carrier-phase recovery: per pilot the correlation summed over both polarizations, 3-pilot average, unwrap, linear interpolation over the frame, derotation; detached from the gradient. `genie_cpr` is a data-aided upper bound (evaluation only) |
| `agc.py` | `agc` | unit mean power per polarization of each image |

No adaptive equalizer, clock recovery or hard decision: the decoder reads the soft symbols. Polarization
demultiplexing is ideal (see `docs/system_overview.md`).
