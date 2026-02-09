const fs = require('fs');
const path = require('path');

const slidesDir = '_slides';
const publicDir = 'public';
const assignmentsDir = 'assignments';
const exercisesDir = 'exercises';

// 读取所有 slides .md 文件 (排除 README.md)
const files = fs.readdirSync(slidesDir).filter(f => f.endsWith('.md') && f !== 'README.md');

// 提取每个文件的标题
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

// 定义每个lecture对应的练习和作业
const lectureResources = {
  1: {
    assignments: []
  },
  2: {
    inClassExercise: {
      title: 'Regularization Problem',
      file: 'Lecture02-Regularization-Problem.md',
      type: 'exercise'
    },
    assignments: ['A1']  // A1 在 Lecture 2 发布
  },
  3: {
    assignments: []
  },
  4: {
    inClassExercise: {
      title: 'Cross-Validation Problem',
      file: 'Lecture04-Cross-Validation-Problem.md',
      type: 'exercise'
    },
    assignments: []
  },
  5: { assignments: [] },
  6: { assignments: [] },
  7: { assignments: [] },
  8: { assignments: [] },
  9: { assignments: [] },
  10: { assignments: [] }
};

// 作业信息
const assignmentInfo = {
  'A0': {
    title: 'A0: Python Warmup',
    description: 'Environment setup and basic NumPy',
    due: 'Week 1',
    weight: '0-2%'
  },
  'A1': {
    title: 'A1: Hedonic Pricing',
    description: 'OLS vs Ridge/Lasso regression',
    due: 'Week 4',
    weight: '10%'
  }
};

// 生成 index.html
const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ECON6083 Machine Learning in Economics | Course Materials</title>
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
      max-width: 1400px;
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

    .lectures-grid {
      display: grid;
      gap: 30px;
      margin-bottom: 60px;
    }

    .lecture-row {
      background: rgba(255, 255, 255, 0.95);
      border-radius: 16px;
      padding: 30px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 30px;
      align-items: start;
    }

    .lecture-left {
      border-right: 2px solid #e2e8f0;
      padding-right: 30px;
    }

    .lecture-right {
      padding-left: 30px;
    }

    .lecture-header {
      display: flex;
      align-items: center;
      gap: 20px;
      margin-bottom: 20px;
    }

    .lecture-number {
      display: flex;
      align-items: center;
      justify-content: center;
      min-width: 60px;
      height: 60px;
      border-radius: 12px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      font-weight: 700;
      font-size: 1.5em;
      flex-shrink: 0;
    }

    .lecture-info {
      flex: 1;
    }

    .lecture-title {
      font-size: 1.3em;
      font-weight: 600;
      color: #2d3748;
      margin-bottom: 5px;
    }

    .slide-link {
      display: inline-block;
      margin-top: 10px;
      padding: 10px 24px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      text-decoration: none;
      border-radius: 8px;
      font-weight: 600;
      transition: all 0.3s ease;
      font-size: 0.95em;
    }

    .slide-link:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .resources-section {
      margin-top: 15px;
    }

    .section-title {
      font-size: 1.1em;
      font-weight: 600;
      color: #4a5568;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .section-title::before {
      content: '';
      width: 4px;
      height: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border-radius: 2px;
    }

    .resource-item {
      background: #f8fafc;
      padding: 15px 20px;
      border-radius: 10px;
      margin-bottom: 10px;
      border-left: 3px solid #667eea;
      transition: all 0.2s ease;
    }

    .resource-item:hover {
      background: #edf2f7;
      transform: translateX(3px);
    }

    .resource-link {
      text-decoration: none;
      color: #2d3748;
      display: block;
    }

    .resource-title {
      font-weight: 600;
      color: #667eea;
      margin-bottom: 5px;
      font-size: 1.05em;
    }

    .resource-desc {
      font-size: 0.9em;
      color: #718096;
    }

    .resource-meta {
      display: flex;
      gap: 15px;
      margin-top: 8px;
      font-size: 0.85em;
    }

    .meta-item {
      display: flex;
      align-items: center;
      gap: 5px;
      color: #718096;
    }

    .badge {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 0.75em;
      font-weight: 600;
      text-transform: uppercase;
    }

    .badge-exercise {
      background: #e0f2fe;
      color: #0369a1;
    }

    .badge-assignment {
      background: #fef3c7;
      color: #92400e;
    }

    .empty-state {
      text-align: center;
      color: #a0aec0;
      font-size: 0.95em;
      padding: 20px;
      font-style: italic;
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

    @media (max-width: 1024px) {
      .lecture-row {
        grid-template-columns: 1fr;
      }

      .lecture-left {
        border-right: none;
        border-bottom: 2px solid #e2e8f0;
        padding-right: 0;
        padding-bottom: 20px;
      }

      .lecture-right {
        padding-left: 0;
        padding-top: 20px;
      }
    }

    @media (max-width: 768px) {
      h1 {
        font-size: 2.5em;
      }

      .hero {
        padding: 40px 24px;
      }

      .lecture-row {
        padding: 20px;
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
            <div class="stat-number">${presentations.length}</div>
            <div class="stat-label">Weeks</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">${presentations.length * 3}</div>
            <div class="stat-label">Hours</div>
          </div>
        </div>
        <span class="course-code">ECON6083</span>
      </div>
    </header>

    <div class="lectures-grid">
      ${presentations.map((p, index) => {
        const lectureNum = p.lectureNum;
        const resources = lectureResources[lectureNum] || {};

        let resourcesHTML = '';

        // In-class exercise
        if (resources.inClassExercise) {
          const ex = resources.inClassExercise;
          resourcesHTML += `
            <div class="resource-item">
              <a href="${exercisesDir}/${ex.file}" class="resource-link" target="_blank">
                <div class="resource-title">
                  <span class="badge badge-exercise">In-Class</span>
                  ${ex.title}
                </div>
                <div class="resource-desc">Practice problem for this lecture</div>
              </a>
            </div>
          `;
        }

        // Assignments
        if (resources.assignments && resources.assignments.length > 0) {
          resources.assignments.forEach(assignmentKey => {
            const assignment = assignmentInfo[assignmentKey];
            if (assignment) {
              resourcesHTML += `
                <div class="resource-item">
                  <a href="${assignmentsDir}/${assignmentKey}/README.md" class="resource-link" target="_blank">
                    <div class="resource-title">
                      <span class="badge badge-assignment">Assignment</span>
                      ${assignment.title}
                    </div>
                    <div class="resource-desc">${assignment.description}</div>
                    <div class="resource-meta">
                      <span class="meta-item">📅 Due: ${assignment.due}</span>
                      <span class="meta-item">📊 Weight: ${assignment.weight}</span>
                    </div>
                  </a>
                </div>
              `;
            }
          });
        }

        if (!resourcesHTML) {
          resourcesHTML = '<div class="empty-state">No exercises or assignments for this lecture</div>';
        }

        return `
        <div class="lecture-row">
          <div class="lecture-left">
            <div class="lecture-header">
              <div class="lecture-number">${String(lectureNum).padStart(2, '0')}</div>
              <div class="lecture-info">
                <div class="lecture-title">${p.title}</div>
              </div>
            </div>
            <a href="${p.htmlName}" class="slide-link" target="_blank">
              📊 View Slides
            </a>
          </div>
          <div class="lecture-right">
            <div class="resources-section">
              <div class="section-title">📝 Exercises & Assignments</div>
              ${resourcesHTML}
            </div>
          </div>
        </div>
        `;
      }).join('\n')}
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
console.log(`✅ Generated index.html with ${presentations.length} lecture(s).`);
