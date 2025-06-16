
import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Loader2, Image } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';

interface ImageSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectImage: (imagePath: string) => void;
  loading: boolean;
}

const ImageSelector = ({ isOpen, onClose, onSelectImage, loading }: ImageSelectorProps) => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  
  // Generate 50 image paths
  const generateImagePaths = () => {
    return Array.from({ length: 50 }, (_, i) => `/images/hazy_${i + 1}.jpg`);
  };
  
  const imagePaths = generateImagePaths();

  const handleImageClick = (imagePath: string) => {
    setSelectedImage(imagePath);
  };

  const handleSelectClick = () => {
    if (selectedImage) {
      onSelectImage(selectedImage);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Image className="h-5 w-5" />
            Select a hazy image to process
          </DialogTitle>
        </DialogHeader>
        
        <ScrollArea className="h-[60vh] pr-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 p-1">
            {imagePaths.map((path, index) => (
              <div 
                key={index} 
                className={`
                  relative rounded-md overflow-hidden border-2 cursor-pointer transition-all hover:opacity-90
                  ${selectedImage === path ? 'border-primary shadow-lg scale-[1.02]' : 'border-transparent hover:border-gray-200'}
                `}
                onClick={() => handleImageClick(path)}
              >
                <div className="aspect-square">
                  <img 
                    src={path} 
                    alt={`Hazy image ${index + 1}`}
                    className="w-full h-full object-cover" 
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = 'https://via.placeholder.com/150?text=Hazy+Image';
                    }}
                  />
                </div>
                {selectedImage === path && (
                  <div className="absolute inset-0 bg-primary/10 flex items-center justify-center">
                    <div className="bg-primary text-xs text-white px-2 py-1 rounded-full">
                      Selected
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </ScrollArea>
        
        <div className="flex justify-end gap-2 mt-4">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button 
            onClick={handleSelectClick} 
            disabled={!selectedImage || loading}
            className="min-w-[140px]"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              'Select & Process'
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ImageSelector;
