const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

const publicDir = 'public';
const assignmentsDir = 'assignments';

// 配置 marked 选项
marked.setOptions({
  breaks: true,
  gfm: true,
  headerIds: true
});

// 生成作业 HTML 页面
function generateAssignmentPage(assignmentKey, readmePath) {
  const markdown = fs.readFileSync(readmePath, 'utf-8');
  const htmlContent = marked.parse(markdown);

  const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${assignmentKey} - ECON6083</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 40px 20px;
      color: #1a202c;
    }

    .container {
      max-width: 900px;
      margin: 0 auto;
      background: white;
      border-radius: 16px;
      padding: 60px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }

    .back-link {
      display: inline-block;
      margin-bottom: 30px;
      padding: 10px 20px;
      background: #667eea;
      color: white !important;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 600;
      transition: all 0.3s;
    }

    .back-link:hover {
      background: #764ba2;
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .content {
      line-height: 1.8;
    }

    .content h1 {
      font-size: 2.5em;
      color: #667eea;
      margin: 30px 0 20px 0;
      border-bottom: 3px solid #667eea;
      padding-bottom: 15px;
    }

    .content h2 {
      font-size: 1.8em;
      color: #4a5568;
      margin-top: 40px;
      margin-bottom: 20px;
      border-left: 4px solid #667eea;
      padding-left: 15px;
    }

    .content h3 {
      font-size: 1.4em;
      color: #2d3748;
      margin-top: 30px;
      margin-bottom: 15px;
    }

    .content h4 {
      font-size: 1.2em;
      color: #4a5568;
      margin-top: 25px;
      margin-bottom: 12px;
    }

    .content p {
      margin-bottom: 15px;
      color: #4a5568;
      line-height: 1.8;
    }

    .content ul, .content ol {
      margin-left: 30px;
      margin-bottom: 20px;
    }

    .content li {
      margin-bottom: 10px;
      color: #4a5568;
      line-height: 1.7;
    }

    .content code {
      background: #f7fafc;
      padding: 3px 8px;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.9em;
      color: #667eea;
      border: 1px solid #e2e8f0;
    }

    .content pre {
      background: #2d3748;
      padding: 20px;
      border-radius: 8px;
      overflow-x: auto;
      margin: 20px 0;
      border-left: 4px solid #667eea;
    }

    .content pre code {
      background: transparent;
      color: #e2e8f0;
      border: none;
      padding: 0;
      font-size: 0.95em;
    }

    .content a {
      color: #667eea;
      text-decoration: none;
      border-bottom: 1px solid #667eea;
      transition: all 0.2s;
    }

    .content a:hover {
      color: #764ba2;
      border-bottom-color: #764ba2;
    }

    .content strong {
      color: #2d3748;
      font-weight: 600;
    }

    .content em {
      color: #718096;
      font-style: italic;
    }

    .content blockquote {
      border-left: 4px solid #667eea;
      padding-left: 20px;
      margin: 20px 0;
      color: #718096;
      font-style: italic;
      background: #f7fafc;
      padding: 15px 20px;
      border-radius: 4px;
    }

    .content table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .content table th {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 12px;
      text-align: left;
      font-weight: 600;
    }

    .content table td {
      padding: 12px;
      border-bottom: 1px solid #e2e8f0;
    }

    .content table tr:nth-child(even) {
      background: #f7fafc;
    }

    .content table tr:hover {
      background: #edf2f7;
    }

    .content hr {
      border: none;
      border-top: 2px solid #e2e8f0;
      margin: 40px 0;
    }

    .footer {
      margin-top: 60px;
      padding-top: 30px;
      border-top: 2px solid #e2e8f0;
      text-align: center;
      color: #a0aec0;
    }

    @media (max-width: 768px) {
      .container {
        padding: 30px 20px;
      }

      .content h1 {
        font-size: 2em;
      }

      .content h2 {
        font-size: 1.5em;
      }

      .content pre {
        padding: 15px;
        font-size: 0.9em;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <a href="../../index.html" class="back-link">← Back to Course Materials</a>
    <div class="content">
      ${htmlContent}
    </div>
    <div class="footer">
      <p>
        <strong>ECON6083</strong> | Machine Learning in Economics | 2026 Spring
      </p>
    </div>
  </div>
</body>
</html>`;

  return html;
}

// 处理所有作业
['A0', 'A1'].forEach(assignmentKey => {
  const readmePath = path.join(assignmentsDir, assignmentKey, 'README.md');
  const outputPath = path.join(publicDir, 'assignments', assignmentKey, 'index.html');

  if (fs.existsSync(readmePath)) {
    const html = generateAssignmentPage(assignmentKey, readmePath);
    fs.writeFileSync(outputPath, html);
    console.log(`✅ Generated ${assignmentKey}/index.html`);
  } else {
    console.log(`⚠️  ${readmePath} not found`);
  }
});

// 处理练习文件
const exercisesDir = 'exercises';
const exerciseFiles = fs.existsSync(exercisesDir) ? fs.readdirSync(exercisesDir).filter(f => f.endsWith('.md')) : [];

exerciseFiles.forEach(file => {
  const exercisePath = path.join(exercisesDir, file);
  const markdown = fs.readFileSync(exercisePath, 'utf-8');
  const htmlContent = marked.parse(markdown);

  const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${file.replace('.md', '')} - ECON6083</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 40px 20px;
      color: #1a202c;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: white;
      border-radius: 16px;
      padding: 60px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    .back-link {
      display: inline-block;
      margin-bottom: 30px;
      padding: 10px 20px;
      background: #667eea;
      color: white !important;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 600;
      transition: all 0.3s;
    }
    .back-link:hover { background: #764ba2; transform: translateY(-2px); }
    .content { line-height: 1.8; }
    .content h1 { font-size: 2.5em; color: #667eea; margin: 30px 0 20px 0; border-bottom: 3px solid #667eea; padding-bottom: 15px; }
    .content h2 { font-size: 1.8em; color: #4a5568; margin-top: 40px; margin-bottom: 20px; border-left: 4px solid #667eea; padding-left: 15px; }
    .content h3 { font-size: 1.4em; color: #2d3748; margin-top: 30px; margin-bottom: 15px; }
    .content p { margin-bottom: 15px; color: #4a5568; line-height: 1.8; }
    .content ul, .content ol { margin-left: 30px; margin-bottom: 20px; }
    .content li { margin-bottom: 10px; color: #4a5568; line-height: 1.7; }
    .content code { background: #f7fafc; padding: 3px 8px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.9em; color: #667eea; border: 1px solid #e2e8f0; }
    .content pre { background: #2d3748; padding: 20px; border-radius: 8px; overflow-x: auto; margin: 20px 0; border-left: 4px solid #667eea; }
    .content pre code { background: transparent; color: #e2e8f0; border: none; padding: 0; }
    .content a { color: #667eea; text-decoration: none; border-bottom: 1px solid #667eea; }
    .content a:hover { color: #764ba2; border-bottom-color: #764ba2; }
    .content strong { color: #2d3748; font-weight: 600; }
    .content blockquote { border-left: 4px solid #667eea; padding: 15px 20px; margin: 20px 0; color: #718096; background: #f7fafc; border-radius: 4px; }
    .content table { width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); }
    .content table th { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; text-align: left; font-weight: 600; }
    .content table td { padding: 12px; border-bottom: 1px solid #e2e8f0; }
    .content table tr:nth-child(even) { background: #f7fafc; }
    .content hr { border: none; border-top: 2px solid #e2e8f0; margin: 40px 0; }
    @media (max-width: 768px) {
      .container { padding: 30px 20px; }
      .content h1 { font-size: 2em; }
      .content h2 { font-size: 1.5em; }
    }
  </style>
</head>
<body>
  <div class="container">
    <a href="../index.html" class="back-link">← Back to Course Materials</a>
    <div class="content">
      ${htmlContent}
    </div>
  </div>
</body>
</html>`;

  const outputPath = path.join(publicDir, 'exercises', file.replace('.md', '.html'));
  fs.writeFileSync(outputPath, html);
  console.log(`✅ Generated exercises/${file.replace('.md', '.html')}`);
});

console.log('✅ All assignment and exercise HTML pages generated');
