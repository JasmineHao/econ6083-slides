const fs = require('fs');
const path = require('path');

const sourceDir = path.join('_slides', 'images');
const targetDir = path.join('public', 'images');

// Function to copy directory recursively
function copyDir(src, dest) {
  // Create destination directory if it doesn't exist
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  // Read all files/folders in source directory
  const entries = fs.readdirSync(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      // Recursively copy subdirectories
      copyDir(srcPath, destPath);
    } else {
      // Copy file
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

// Check if source images directory exists
if (fs.existsSync(sourceDir)) {
  console.log('Copying images from _slides/images to public/images...');
  copyDir(sourceDir, targetDir);
  console.log('✅ Images copied successfully!');
} else {
  console.log('⚠️  No images directory found in _slides/');
}
