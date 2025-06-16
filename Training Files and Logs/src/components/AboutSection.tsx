
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const AboutSection = () => {
  return (
    <section id="about" className="py-20 bg-secondary/50">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl font-bold mb-12 text-center">About DeHaze Technology</h2>
        
        <div className="grid md:grid-cols-3 gap-8">
          <Card>
            <CardHeader>
              <CardTitle>What is Image Dehazing?</CardTitle>
            </CardHeader>
            <CardContent>
              <p>
                Image dehazing is the process of removing haze, fog, smoke, or mist from images to improve clarity
                and visibility. Our system uses advanced deep learning algorithms to restore the original colors
                and details of the scene.
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>How It Works</CardTitle>
            </CardHeader>
            <CardContent>
              <p>
                Our AI model has been trained on thousands of pairs of hazy and clear images. It can identify
                the atmospheric light and estimate the transmission map to recover a haze-free image with
                enhanced visibility and natural colors.
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Applications</CardTitle>
            </CardHeader>
            <CardContent>
              <p>
                Image dehazing technology has practical applications in photography, surveillance systems,
                outdoor imaging, autonomous vehicles, and any vision system that needs to function effectively in
                adverse weather conditions.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;
