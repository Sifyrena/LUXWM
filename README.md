# LUXWM
FWPhys LUX’s own command-line photo watermarking tool making use of EXIF data. 

## Watermark files
place your own watermark files in ./Private. For dark mode, _MarkW.png, and for bright mode, _MarkD.png.

## Basic usage
```python 
import LUXWM
LUXWM.Process_Batch('Inputs/', Silent = True, SeriesTitle= "", Idempotence = True)
```

