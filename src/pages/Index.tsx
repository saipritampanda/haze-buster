
import React, { useState } from 'react';
import Header from '@/components/Header';
import Hero from '@/components/Hero';
import ImageSelector from '@/components/ImageSelector';
import ImageComparison from '@/components/ImageComparison';
import AboutSection from '@/components/AboutSection';
import Footer from '@/components/Footer';
import { toast } from '@/components/ui/sonner';
import { processSampleImage, dehazeImage } from '@/services/dehazeService';

const Index = () => {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);

  const handleDialogOpen = () => {
    setIsDialogOpen(true);
  };

  const handleDialogClose = () => {
    setIsDialogOpen(false);
  };

  const handleSelectImage = async (imagePath: string) => {
    setLoading(true);
    setOriginalImage(imagePath);
    
    try {
      // Call the processSampleImage function from dehazeService
      const result = await processSampleImage(imagePath);
      
      if (result.success && result.imageUrl) {
        setProcessedImage(result.imageUrl);
        toast.success("Image successfully dehazed!");
      } else {
        toast.error(result.error || "Error processing image");
        setProcessedImage(null);
      }
      handleDialogClose();
    } catch (error) {
      toast.error("Error processing image. Please try again.");
      console.error("Error processing image:", error);
      setProcessedImage(null);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setLoading(true);
    
    // Create object URL for the original image preview
    const originalUrl = URL.createObjectURL(file);
    setOriginalImage(originalUrl);
    
    try {
      // Process the uploaded file
      const result = await dehazeImage(file);
      
      if (result.success && result.imageUrl) {
        setProcessedImage(result.imageUrl);
        toast.success("Image successfully dehazed!");
      } else {
        toast.error(result.error || "Error processing image");
        setProcessedImage(null);
      }
    } catch (error) {
      toast.error("Error processing image. Please try again.");
      console.error("Error processing uploaded file:", error);
      setProcessedImage(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-grow">
        <Hero 
          onOpenDialog={handleDialogOpen}
          onFileUpload={handleFileUpload}
        />
        
        <ImageSelector 
          isOpen={isDialogOpen}
          onClose={handleDialogClose}
          onSelectImage={handleSelectImage}
          loading={loading}
        />
        
        {originalImage && processedImage && (
          <ImageComparison 
            originalImage={originalImage} 
            processedImage={processedImage} 
          />
        )}
        
        <AboutSection />
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
