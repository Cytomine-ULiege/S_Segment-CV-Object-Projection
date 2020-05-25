FROM cytomineuliege/software-python3-base:v2.7.0-py3.7.6

RUN pip install sldc sldc-cytomine rasterio scikit-image

ADD run.py /app/run.py
ADD mask_to_polygons.py /app/mask_to_polygons.py
ADD sldc_adapter.py /app/sldc_adapter.py

ENTRYPOINT ["python", "/app/run.py"]