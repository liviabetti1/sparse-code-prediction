
1. Download datasets:
    bash: cd datasets

    EuroSAT: python EuroSAT.py --download --root /data/

    For the MOSAIKS datasets, I will provide them on google drive (can be downloaded on code ocean)

    Make sure to update paths on datasets/paths.yaml

2. GeoCLIP and SatCLIP are implemented. For GeoCLIP:
need to pip install geoclip
for satclip, need to clone the repo in external/ (to use the get_satclip function)
(for now, I'll just push this. I remember having to make some changes to the imports for this to work so this will be easier for now.)

Note: I provided a location_encoder.py model that can be used to load satclip and geoclip. However, note that if the location model is satclip, I internally switch to the ordering lon, lat (instead of lat, lon for geoclip) so make sure not to do this yourself if using my model.