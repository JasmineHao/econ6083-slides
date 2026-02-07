const fs = require('fs');
const path = require('path');

const slidesDir = '_slides';
const publicDir = 'public';

// 读取所有 .md 文件
const files = fs.readdirSync(slidesDir).filter(f => f.endsWith('.md'));

// 提取每个文件的标题（第一个 # 标题）
const presentations = files.map(f => {
  const content = fs.readFileSync(path.join(slidesDir, f), 'utf-8');
  const titleMatch = content.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1] : f.replace('.md', '');
  const htmlName = f.replace('.md', '.html');
  return { title, htmlName, filename: f };
});

// 生成 index.html
const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ECON6083 Presentations</title>
  <style>
    body {
      font-family: 'Segoe UI', 'Georgia', serif;
      max-width: 900px;
      margin: 60px auto;
      padding: 0 30px;
      color: #1e293b;
      background: linear-gradient(to bottom, #ffffff 0%, #f8fafc 100%);
      min-height: 100vh;
    }
    h1 {
      color: #0f172a;
      border-bottom: 4px solid #3b82f6;
      padding-bottom: 15px;
      font-size: 2.5em;
    }
    .subtitle {
      color: #64748b;
      font-size: 1.2em;
      margin-top: -10px;
      margin-bottom: 40px;
    }
    ul {
      list-style: none;
      padding: 0;
    }
    li {
      margin: 20px 0;
    }
    a {
      color: #1e40af;
      text-decoration: none;
      font-size: 1.3em;
      padding: 16px 24px;
      display: block;
      border-radius: 8px;
      transition: all 0.2s;
      background: white;
      border: 2px solid #e2e8f0;
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    a:hover {
      background: #eff6ff;
      border-color: #3b82f6;
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
      transform: translateY(-2px);
    }
    .lecture-number {
      color: #3b82f6;
      font-weight: 600;
      margin-right: 10px;
    }
    footer {
      margin-top: 60px;
      padding-top: 20px;
      border-top: 1px solid #e2e8f0;
      text-align: center;
      color: #64748b;
      font-size: 0.9em;
    }
  </style>
</head>
<body>
  <h1>📊 ECON6083 Presentations</h1>
  <p class="subtitle">Machine Learning in Economics - Course Slides</p>
  <ul>
    ${presentations.map((p, index) => `<li><a href="${p.htmlName}"><span class="lecture-number">Lecture ${index + 1}:</span>${p.title}</a></li>`).join('\n    ')}
  </ul>
  <footer>
    <p>Generated with <a href="https://marp.app/" target="_blank">Marp</a> | ECON6083 Course Materials</p>
  </footer>
</body>
</html>`;

if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir);
fs.writeFileSync(path.join(publicDir, 'index.html'), html);
console.log(`✅ Generated index.html with ${presentations.length} presentation(s).`);
