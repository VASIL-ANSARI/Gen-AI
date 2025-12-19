from google import genai
from google.genai import types
import os
import io
from PIL import Image
import google.auth

def build_vertex_client():
    credentials, project_id = google.auth.default()
    
    return genai.Client(
        vertexai=True,
        credentials=credentials,
        project=project_id,
        location="us-central1"
    )

def generate(query):
    client = build_vertex_client()

    model = "gemini-2.5-flash-image"

    contents = [
        types.Content(
            role="user",
            parts=[
                {
                    "text": query
                }
            ]
        )
    ]

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            image_size="1K",
            aspect_ratio="1:1",
            output_mime_type="image/png",
        ),
    )

    image_byte_chunks = []

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if not chunk.candidates:
            continue

        for part in chunk.candidates[0].content.parts:
            # # TEXT
            # if part.text:
            #     print(part.text, end="")

            # IMAGE (already raw bytes)
            if part.inline_data:
                image_byte_chunks.append(part.inline_data.data)

    # print(image_byte_chunks)
    # ✅ Combine AFTER streaming finishes
    if image_byte_chunks:
        image_bytes = b"".join(image_byte_chunks)

        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")

        # image.save("output.png", "PNG")

        return image
    else:
        return None
    
        # image.save("output.jpg", "JPEG", quality=95)

        # print("\n[Image saved as output.png and output.jpg]")

# generate("An illustration showing a MacBook with a Windows virtual machine interface open, displaying the Power BI application.")