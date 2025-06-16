
export interface DehazeResult {
  success: boolean;
  message?: string;
  imageUrl?: string;
  error?: string;
}

// Backend URL - updated to point to Render deployment
const BACKEND_URL = "https://haze-buster-image-revive.onrender.com";

export const dehazeImage = async (imageFile: File): Promise<DehazeResult> => {
  try {
    // Create form data to send the image file
    const formData = new FormData();
    formData.append('file', imageFile);

    // Call the Python backend dehaze endpoint
    const response = await fetch(`${BACKEND_URL}/dehaze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      console.error("Error from backend:", errorData);
      return {
        success: false,
        error: errorData.detail || "Failed to process image"
      };
    }

    // Get the image directly as blob from the response
    const imageBlob = await response.blob();
    const imageUrl = URL.createObjectURL(imageBlob);

    return {
      success: true,
      imageUrl,
      message: "Image successfully dehazed"
    };
  } catch (error) {
    console.error("Error in dehazeImage service:", error);
    return {
      success: false,
      error: "An unexpected error occurred"
    };
  }
};

// Function to handle processing of sample images
export const processSampleImage = async (imagePath: string): Promise<DehazeResult> => {
  try {
    console.log("Processing sample image:", imagePath);
    
    // Fetch the sample image first 
    const imageResponse = await fetch(imagePath);
    if (!imageResponse.ok) {
      return {
        success: false,
        error: "Failed to fetch the sample image"
      };
    }
    
    const imageBlob = await imageResponse.blob();
    
    // Create a File object from the blob
    const imageFile = new File([imageBlob], imagePath.split('/').pop() || 'sample.jpg', { 
      type: imageBlob.type 
    });
    
    // Use the same dehazeImage function to process the sample
    return await dehazeImage(imageFile);
  } catch (error) {
    console.error("Error in processSampleImage service:", error);
    return {
      success: false,
      error: "An unexpected error occurred"
    };
  }
};
