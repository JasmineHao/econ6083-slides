const fs = require('fs');
const path = require('path');

const publicDir = 'public';

// 复制文件夹的递归函数
function copyFolderRecursive(source, target) {
  // 创建目标文件夹
  if (!fs.existsSync(target)) {
    fs.mkdirSync(target, { recursive: true });
  }

  // 读取源文件夹
  const files = fs.readdirSync(source);

  files.forEach(file => {
    const sourcePath = path.join(source, file);
    const targetPath = path.join(target, file);
    const stat = fs.statSync(sourcePath);

    if (stat.isDirectory()) {
      // 跳过 .git 文件夹
      if (file === '.git' || file === 'node_modules') {
        return;
      }
      // 递归复制子文件夹
      copyFolderRecursive(sourcePath, targetPath);
    } else {
      // 复制文件
      fs.copyFileSync(sourcePath, targetPath);
    }
  });
}

// 确保 public 目录存在
if (!fs.existsSync(publicDir)) {
  fs.mkdirSync(publicDir);
}

// 复制 assignments 文件夹
if (fs.existsSync('assignments')) {
  console.log('📁 Copying assignments/ to public/assignments/...');
  copyFolderRecursive('assignments', path.join(publicDir, 'assignments'));
  console.log('✅ Assignments copied');
} else {
  console.log('⚠️  assignments/ folder not found');
}

// 复制 exercises 文件夹
if (fs.existsSync('exercises')) {
  console.log('📁 Copying exercises/ to public/exercises/...');
  copyFolderRecursive('exercises', path.join(publicDir, 'exercises'));
  console.log('✅ Exercises copied');
} else {
  console.log('⚠️  exercises/ folder not found');
}

console.log('✅ All resources copied to public/');
