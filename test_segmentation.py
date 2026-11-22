
from rembg import remove, new_session
from PIL import Image
import io

# Initialize segmentation model
session = new_session("isnet-general-use")

# Load test image
image = Image.open("test_leaf.jpg").convert("RGB")

# Apply segmentation
result = remove(image, session=session)

# Save segmented result
result.save("segmented_leaf.png")

print("Segmentation completed!")
print("Saved: segmented_leaf.png")
