
import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-background border-t py-8">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <div className="flex items-center gap-2">
              <div className="bg-primary w-8 h-8 rounded-full flex items-center justify-center">
                <span className="text-primary-foreground font-bold">D</span>
              </div>
              <span className="font-bold">DeHaze</span>
            </div>
            <p className="text-sm text-muted-foreground mt-2">AI-powered image dehazing solution</p>
          </div>
          
          <div className="text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} DeHaze. All rights reserved.
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
