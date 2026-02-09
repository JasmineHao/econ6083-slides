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

  // Extract lecture number from filename
  const numMatch = f.match(/Lecture[- ]?(\d+)/i);
  const lectureNum = numMatch ? parseInt(numMatch[1], 10) : 999;

  return { title, htmlName, filename: f, lectureNum };
});

// Sort by lecture number
presentations.sort((a, b) => a.lectureNum - b.lectureNum);

// 生成 index.html
const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ECON6083 Machine Learning in Economics | Course Slides</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 40px 20px;
      color: #1a202c;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
    }

    header {
      text-align: center;
      margin-bottom: 60px;
      animation: fadeInDown 0.8s ease-out;
    }

    .hero {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      border-radius: 24px;
      padding: 60px 40px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
      margin-bottom: 40px;
    }

    h1 {
      font-size: 3.5em;
      font-weight: 800;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 20px;
      letter-spacing: -0.02em;
    }

    .subtitle {
      font-size: 1.4em;
      color: #4a5568;
      font-weight: 500;
      margin-bottom: 10px;
    }

    .course-code {
      display: inline-block;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 8px 24px;
      border-radius: 50px;
      font-size: 0.9em;
      font-weight: 600;
      margin-top: 20px;
      letter-spacing: 1px;
    }

    .lectures-list {
      background: rgba(255, 255, 255, 0.95);
      border-radius: 16px;
      padding: 40px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
      margin-bottom: 40px;
      max-width: 900px;
      margin-left: auto;
      margin-right: auto;
    }

    .lectures-list ol {
      list-style: none;
      counter-reset: lecture-counter;
      padding: 0;
      margin: 0;
    }

    .lecture-item {
      counter-increment: lecture-counter;
      margin-bottom: 16px;
      padding: 20px;
      border-radius: 12px;
      background: #f8fafc;
      transition: all 0.3s ease;
      border-left: 4px solid transparent;
    }

    .lecture-item:hover {
      background: #edf2f7;
      border-left-color: #667eea;
      transform: translateX(4px);
    }

    .lecture-item a {
      display: flex;
      align-items: center;
      text-decoration: none;
      color: inherit;
      gap: 20px;
    }

    .lecture-number {
      display: flex;
      align-items: center;
      justify-content: center;
      min-width: 50px;
      height: 50px;
      border-radius: 10px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      font-weight: 700;
      font-size: 1.2em;
      flex-shrink: 0;
    }

    .lecture-number::before {
      content: counter(lecture-counter, decimal-leading-zero);
    }

    .lecture-title {
      flex: 1;
      font-size: 1.1em;
      font-weight: 500;
      color: #2d3748;
      line-height: 1.5;
    }

    footer {
      text-align: center;
      padding: 40px 20px;
      background: rgba(255, 255, 255, 0.95);
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }

    footer p {
      color: #718096;
      font-size: 0.95em;
      line-height: 1.8;
    }

    footer a {
      color: #667eea;
      text-decoration: none;
      font-weight: 600;
      transition: color 0.2s;
    }

    footer a:hover {
      color: #764ba2;
    }

    .stats {
      display: flex;
      justify-content: center;
      gap: 40px;
      margin-top: 20px;
    }

    .stat-item {
      text-align: center;
    }

    .stat-number {
      font-size: 2.5em;
      font-weight: 800;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .stat-label {
      color: #718096;
      font-size: 0.9em;
      margin-top: 4px;
    }

    @keyframes fadeInDown {
      from {
        opacity: 0;
        transform: translateY(-30px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @keyframes fadeInUp {
      from {
        opacity: 0;
        transform: translateY(30px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @media (max-width: 768px) {
      h1 {
        font-size: 2.5em;
      }

      .hero {
        padding: 40px 24px;
      }

      .lectures-list {
        padding: 20px;
      }

      .lecture-item a {
        gap: 12px;
      }

      .lecture-number {
        min-width: 40px;
        height: 40px;
        font-size: 1em;
      }

      .lecture-title {
        font-size: 0.95em;
      }

      .stats {
        flex-direction: column;
        gap: 20px;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="hero">
        <h1>Machine Learning in Economics</h1>
        <p class="subtitle">Faculty of Business and Economics</p>
        <div class="stats">
          <div class="stat-item">
            <div class="stat-number">${presentations.length}</div>
            <div class="stat-label">Lectures</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">11</div>
            <div class="stat-label">Weeks</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">33</div>
            <div class="stat-label">Hours</div>
          </div>
        </div>
        <span class="course-code">ECON6083</span>
      </div>
    </header>

    <div class="lectures-list">
      <ol>
        ${presentations.map((p, index) => `<li class="lecture-item">
          <a href="${p.htmlName}">
            <div class="lecture-number"></div>
            <div class="lecture-title">${p.title}</div>
          </a>
        </li>`).join('\n        ')}
      </ol>
    </div>

    <footer>
      <p>
        <strong>Contact:</strong> <a href="mailto:haoyu@hku.hk">haoyu@hku.hk</a><br>
        <strong>Course Materials:</strong> <a href="https://jasminehao.com/econ6083-slides/" target="_blank">Website</a><br>
        <br>
        <strong>Course Slides</strong> powered by <a href="https://marp.app/" target="_blank">Marp</a><br>
        ECON6083 | 2026 Spring | The University of Hong Kong
      </p>
    </footer>
  </div>
</body>
</html>`;

if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir);
fs.writeFileSync(path.join(publicDir, 'index.html'), html);
console.log(`✅ Generated index.html with ${presentations.length} presentation(s).`);
