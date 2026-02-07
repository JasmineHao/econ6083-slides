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
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ECON6083 机器学习在经济学中的应用 | 课程讲义</title>
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

    .lectures-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
      gap: 24px;
      margin-bottom: 60px;
    }

    .lecture-card {
      background: rgba(255, 255, 255, 0.95);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      animation: fadeInUp 0.6s ease-out backwards;
      text-decoration: none;
      color: inherit;
      display: block;
      position: relative;
    }

    .lecture-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 6px;
      background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
      transform: scaleX(0);
      transform-origin: left;
      transition: transform 0.3s ease;
    }

    .lecture-card:hover::before {
      transform: scaleX(1);
    }

    .lecture-card:hover {
      transform: translateY(-8px);
      box-shadow: 0 16px 40px rgba(102, 126, 234, 0.4);
    }

    .lecture-number {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 16px 24px;
      font-weight: 700;
      font-size: 0.85em;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .lecture-number::before {
      content: '📚';
      font-size: 1.2em;
    }

    .lecture-content {
      padding: 24px;
    }

    .lecture-title {
      font-size: 1.3em;
      font-weight: 600;
      color: #2d3748;
      line-height: 1.4;
      margin-bottom: 12px;
    }

    .lecture-meta {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 16px;
    }

    .meta-tag {
      background: #edf2f7;
      color: #4a5568;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.75em;
      font-weight: 500;
      font-family: 'JetBrains Mono', monospace;
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

      .lectures-grid {
        grid-template-columns: 1fr;
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
        <h1>机器学习在经济学中的应用</h1>
        <p class="subtitle">Machine Learning in Economics</p>
        <div class="stats">
          <div class="stat-item">
            <div class="stat-number">${presentations.length}</div>
            <div class="stat-label">讲次 Lectures</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">11</div>
            <div class="stat-label">周 Weeks</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">33</div>
            <div class="stat-label">学时 Hours</div>
          </div>
        </div>
        <span class="course-code">ECON6083</span>
      </div>
    </header>

    <div class="lectures-grid">
      ${presentations.map((p, index) => {
        const lectureNum = p.filename.match(/Lecture-(\d+)/i)?.[1] || (index + 1);
        return `<a href="${p.htmlName}" class="lecture-card" style="animation-delay: ${index * 0.1}s">
        <div class="lecture-number">第 ${lectureNum} 讲</div>
        <div class="lecture-content">
          <div class="lecture-title">${p.title}</div>
          <div class="lecture-meta">
            <span class="meta-tag">HTML</span>
            <span class="meta-tag">Slides</span>
          </div>
        </div>
      </a>`;
      }).join('\n      ')}
    </div>

    <footer>
      <p>
        <strong>课程讲义</strong> 由 <a href="https://marp.app/" target="_blank">Marp</a> 强力驱动<br>
        ECON6083 | 2026 Spring | 中国人民大学经济学院
      </p>
    </footer>
  </div>
</body>
</html>`;

if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir);
fs.writeFileSync(path.join(publicDir, 'index.html'), html);
console.log(`✅ Generated index.html with ${presentations.length} presentation(s).`);
