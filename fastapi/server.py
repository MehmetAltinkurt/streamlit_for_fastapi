import io
from segmentation import get_segmentator, get_segments
from starlette.responses import Response

from fastapi import FastAPI, File

model = get_segmentator()

app = FastAPI(
    title="YOLO5 image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via YOLO5 implemented in PyTorch.""",
)


@app.post("/segmentation")
def get_segmentation_map(file: bytes = File(...)):
    """Get segmentation maps from image file"""
    segmented_image = get_segments(model, file)
    bytes_io = io.BytesIO()
    segmented_image.save(bytes_io, format="PNG")
    print("post")
    return Response(bytes_io.getvalue(), media_type="image/png")
