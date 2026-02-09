const fs = require('fs');
const path = require('path');

const publicDir = 'public';
const assignmentsDir = 'assignments';

// 简单的 markdown 到 HTML 转换（基础版本）
function markdownToHtml(markdown) {
  let html = markdown
    // Headers
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    // Bold
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    // Code blocks
    .replace(/```([^`]+)```/gs, '<pre><code>$1</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
    // Lists (simple)
    .replace(/^\- (.+)$/gm, '<li>$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')
    // Paragraphs
    .replace(/\n\n/g, '</p><p>')
    // Blockquotes
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

  return '<p>' + html + '</p>';
}

// 生成作业 HTML 页面
function generateAssignmentPage(assignmentKey, readmePath) {
  const markdown = fs.readFileSync(readmePath, 'utf-8');
  const htmlContent = markdownToHtml(markdown);

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
      line-height: 1.7;
    }

    .container {
      max-width: 900px;
      margin: 0 auto;
      background: white;
      border-radius: 16px;
      padding: 60px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }

    h1 {
      font-size: 2.5em;
      color: #667eea;
      margin-bottom: 20px;
      border-bottom: 3px solid #667eea;
      padding-bottom: 15px;
    }

    h2 {
      font-size: 1.8em;
      color: #4a5568;
      margin-top: 40px;
      margin-bottom: 20px;
      border-left: 4px solid #667eea;
      padding-left: 15px;
    }

    h3 {
      font-size: 1.3em;
      color: #2d3748;
      margin-top: 30px;
      margin-bottom: 15px;
    }

    p {
      margin-bottom: 15px;
      color: #4a5568;
    }

    ul, ol {
      margin-left: 30px;
      margin-bottom: 20px;
    }

    li {
      margin-bottom: 10px;
      color: #4a5568;
    }

    code {
      background: #f7fafc;
      padding: 3px 8px;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.9em;
      color: #667eea;
      border: 1px solid #e2e8f0;
    }

    pre {
      background: #2d3748;
      padding: 20px;
      border-radius: 8px;
      overflow-x: auto;
      margin: 20px 0;
    }

    pre code {
      background: transparent;
      color: #e2e8f0;
      border: none;
      padding: 0;
    }

    a {
      color: #667eea;
      text-decoration: none;
      border-bottom: 1px solid #667eea;
    }

    a:hover {
      color: #764ba2;
      border-bottom-color: #764ba2;
    }

    strong {
      color: #2d3748;
      font-weight: 600;
    }

    blockquote {
      border-left: 4px solid #667eea;
      padding-left: 20px;
      margin: 20px 0;
      color: #718096;
      font-style: italic;
    }

    .back-link {
      display: inline-block;
      margin-bottom: 30px;
      padding: 10px 20px;
      background: #667eea;
      color: white;
      border-radius: 8px;
      text-decoration: none;
      border: none;
    }

    .back-link:hover {
      background: #764ba2;
    }

    .file-list {
      background: #f7fafc;
      padding: 20px;
      border-radius: 8px;
      margin: 20px 0;
    }

    .file-list h3 {
      margin-top: 0;
    }

    @media (max-width: 768px) {
      .container {
        padding: 30px 20px;
      }

      h1 {
        font-size: 2em;
      }

      h2 {
        font-size: 1.5em;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <a href="../../index.html" class="back-link">← Back to Course Materials</a>
    ${htmlContent}
    <div style="margin-top: 60px; padding-top: 30px; border-top: 2px solid #e2e8f0;">
      <p style="text-align: center; color: #a0aec0;">
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
  const htmlContent = markdownToHtml(markdown);

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
      line-height: 1.7;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: white;
      border-radius: 16px;
      padding: 60px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    h1 { font-size: 2.5em; color: #667eea; margin-bottom: 20px; border-bottom: 3px solid #667eea; padding-bottom: 15px; }
    h2 { font-size: 1.8em; color: #4a5568; margin-top: 40px; margin-bottom: 20px; border-left: 4px solid #667eea; padding-left: 15px; }
    h3 { font-size: 1.3em; color: #2d3748; margin-top: 30px; margin-bottom: 15px; }
    p { margin-bottom: 15px; color: #4a5568; }
    code { background: #f7fafc; padding: 3px 8px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.9em; color: #667eea; }
    pre { background: #2d3748; padding: 20px; border-radius: 8px; overflow-x: auto; margin: 20px 0; }
    pre code { background: transparent; color: #e2e8f0; }
    .back-link {
      display: inline-block;
      margin-bottom: 30px;
      padding: 10px 20px;
      background: #667eea;
      color: white;
      border-radius: 8px;
      text-decoration: none;
    }
    .back-link:hover { background: #764ba2; }
  </style>
</head>
<body>
  <div class="container">
    <a href="../index.html" class="back-link">← Back to Course Materials</a>
    ${htmlContent}
  </div>
</body>
</html>`;

  const outputPath = path.join(publicDir, 'exercises', file.replace('.md', '.html'));
  fs.writeFileSync(outputPath, html);
  console.log(`✅ Generated exercises/${file.replace('.md', '.html')}`);
});

console.log('✅ All assignment and exercise HTML pages generated');
