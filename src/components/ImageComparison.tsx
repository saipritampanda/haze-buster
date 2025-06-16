
import React from 'react';
import { Button } from '@/components/ui/button';
import { Download } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

interface ImageComparisonProps {
  originalImage: string;
  processedImage: string;
}

const ImageComparison = ({ originalImage, processedImage }: ImageComparisonProps) => {
  const handleDownload = () => {
    // Create a link and trigger a download
    const link = document.createElement('a');
    link.href = processedImage;
    link.download = 'dehazed_image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="mt-16 mb-16 container mx-auto px-4 animate-fade-in">
      <h2 className="text-2xl font-bold mb-6 text-center">Before & After</h2>
      <div className="image-comparison mb-8">
        <Card>
          <CardContent className="p-4">
            <div className="aspect-square w-full relative">
              <img 
                src={originalImage} 
                alt="Original hazy image" 
                className="max-w-full h-auto object-contain rounded-md"
              />
            </div>
            <p className="text-center mt-3 font-medium">Original Image</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="aspect-square w-full relative">
              <img 
                src={processedImage} 
                alt="Dehazed image" 
                className="max-w-full h-auto object-contain rounded-md"
              />
            </div>
            <p className="text-center mt-3 font-medium">Dehazed Image</p>
          </CardContent>
        </Card>
      </div>
      
      <div className="flex justify-center">
        <Button onClick={handleDownload}>
          <Download className="mr-2 h-4 w-4" />
          Download Dehazed Image
        </Button>
      </div>
    </div>
  );
};

export default ImageComparison;
