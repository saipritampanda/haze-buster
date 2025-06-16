
import React from 'react';

const Header = () => {
  return (
    <header className="w-full py-4 border-b">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="bg-primary w-8 h-8 rounded-full flex items-center justify-center">
            <span className="text-primary-foreground font-bold">D</span>
          </div>
          <h1 className="text-xl font-bold">DeHaze</h1>
        </div>
        <nav>
          <ul className="flex gap-6">
            <li><a href="#" className="text-sm font-medium hover:text-primary transition-colors">Home</a></li>
            <li><a href="#about" className="text-sm font-medium hover:text-primary transition-colors">About</a></li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;
