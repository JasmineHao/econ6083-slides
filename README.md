# ECON6083 Course Presentations

This repository contains all course presentations for ECON6083: Machine Learning in Economics, powered by [Marp](https://marp.app/).

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
npm install

# Start live preview server (opens at http://localhost:8080)
npm start

# Build all slides to HTML
npm run build

# Generate index page
npm run build:index

# Export to PDF (optional)
npm run pdf
```

### Project Structure

```
presentations/
├── _slides/                    # Marp Markdown source files
│   └── Lecture-08-DAGs-and-Causal-Identification.md
├── public/                     # Built HTML/PDF output (generated)
├── themes/                     # Custom Marp themes
│   └── academic.css           # Academic presentation theme
├── scripts/
│   └── build-index.js         # Index page generator
├── .github/workflows/
│   └── deploy.yml             # GitHub Actions auto-deploy
├── package.json
├── .marprc.yml                # Marp CLI configuration
└── README.md
```

## 📝 Creating New Presentations

1. Create a new `.md` file in `_slides/`
2. Add Marp front matter:

```markdown
---
marp: true
theme: academic
paginate: true
math: mathjax
footer: 'ECON6083 Lecture X | Topic'
---

<!-- _class: lead -->

# Lecture Title
## Subtitle
### Author Name
#### Date

---

## First Slide

Content here...
```

3. Build and preview with `npm start`

## 🎨 Theme Customization

Edit `themes/academic.css` to customize the presentation style. The academic theme includes:

- Clean, professional design with gradient backgrounds
- Blue color scheme optimized for academic content
- Special slide classes:
  - `<!-- _class: lead -->` for title slides
  - `<!-- _class: cols -->` for two-column layouts
- Beautiful table styling with gradients and hover effects
- Code syntax highlighting
- MathJax support for equations

## 🌐 Deployment

The presentations are automatically deployed to GitHub Pages when you push to the `main` branch.

### Setup GitHub Pages

1. Go to repository Settings → Pages
2. Under "Source", select "GitHub Actions"
3. Push code to `main` branch
4. Visit `https://username.github.io/repository-name/`

## 📄 Exporting to PDF

```bash
npm run pdf
```

PDFs will be generated in the `public/` directory.

## 🔧 Marp Syntax Reference

- **Slide separator**: `---`
- **Math**: Inline `$...$` or block `$$...$$`
- **Images**: `![description](path/to/image.png)`
- **Special classes**: `<!-- _class: lead -->` or `<!-- _class: cols -->`
- **Two columns**: Use `section.cols` class
- **Code blocks**: `` ```language ... ``` ``

## 📚 Resources

- [Marp Documentation](https://marpit.marp.app/)
- [Marp CLI](https://github.com/marp-team/marp-cli)
- [Markdown Guide](https://www.markdownguide.org/)

## 📧 Contact

For questions about the course content, contact [instructor email].

---

**License**: MIT | **Course**: ECON6083 | **Powered by**: Marp
