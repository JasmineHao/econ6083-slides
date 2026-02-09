# A0: Python Warmup + Course Feedback

> **Assignment Type**: Ungraded (or 1-2% participation credit)
>
> **Purpose**:
> 1. Verify your Python environment works
> 2. Understand your background and expectations to adjust the course
> 3. Familiarize yourself with the submission process

---

## 🎯 Assignment Content

### Part 1: Write Three Simple Functions (30 minutes)

Complete three basic statistical functions in `basic_stats.py`:

1. **`calculate_mean(numbers)`** - Calculate mean
2. **`calculate_std(numbers)`** - Calculate standard deviation
3. **`normalize_array(numbers)`** - Normalize array

These functions are **very simple**, mainly to verify you can:
- Use basic NumPy functions
- Understand function inputs and outputs
- Run unit tests

### Part 2: Fill Out Feedback Questionnaire (10 minutes)

Complete `feedback.md` with:
- Your programming background
- Expectations for the course
- Support you need

**Important**: The questionnaire does not affect your grade, please answer honestly!

---

## 🚀 Getting Started

### 1. Download Assignment Files

Download the assignment package (A0.zip) from the Moodle course page and extract it to your working directory.

### 2. Install Dependencies

Open terminal (Windows: use PowerShell or CMD), navigate to the assignment folder:

```bash
cd path/to/A0

# Install NumPy and pytest
pip install numpy pytest

# Or use requirements.txt
pip install -r requirements.txt
```

### 3. Complete the Code

Open `basic_stats.py`, find the sections marked with `# TODO`, and fill in your code.

**Hints**:
```python
# Calculate mean
def calculate_mean(numbers):
    return np.mean(numbers)  # or sum(numbers) / len(numbers)

# Calculate standard deviation
def calculate_std(numbers):
    return np.std(numbers)

# Normalize
def normalize_array(numbers):
    numbers = np.array(numbers)
    mean = np.mean(numbers)
    std = np.std(numbers)
    return (numbers - mean) / std
```

### 4. Test Your Code

```bash
# Run your own tests
python basic_stats.py

# Run automated tests
pytest test_basic_stats.py -v
```

If all tests pass, your code is correct!

### 5. Fill Out Feedback Questionnaire

Open `feedback.md` and answer the questions.

### 6. Submit Assignment

Package the following files into a zip file:
- `basic_stats.py` (your completed code)
- `feedback.md` (completed questionnaire)

**Naming Format**: `A0_YourStudentID_YourLastName.zip`

Example: `A0_2024001_Smith.zip`

Then upload this zip file to Moodle.

---

## ✅ Grading Criteria (if graded)

| Component | Weight | Description |
|-----------|--------|-------------|
| Code Correctness | 50% | Pass automated tests |
| Feedback Questionnaire | 50% | Complete submission |

**But actually**: This assignment is mainly to familiarize you with the process - you get credit just for submitting!

---

## 🆘 Having Problems?

### Common Questions

**Q: I don't know how to install Python, what should I do?**

A:
1. Python: Visit [python.org](https://python.org), download version 3.8+
2. Check "Add Python to PATH" during installation
3. Or come to office hours, we'll solve it together!

**Q: Tests are failing, what should I do?**

A:
1. Read the error message - it usually tells you what went wrong
2. Use `print()` to output intermediate results and debug your code
3. Check if you're returning the correct type (e.g., should return float, not None)
4. Come to office hours for help

**Q: I don't know NumPy at all, what should I do?**

A:
- These three functions only need `np.mean()`, `np.std()` and basic arithmetic
- Check out [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- Or just use pure Python: `sum(numbers) / len(numbers)`

**Q: How do I create a zip file on Windows?**

A:
1. Select both `basic_stats.py` and `feedback.md` files
2. Right-click → "Send to" → "Compressed (zipped) folder"
3. Rename to `A0_StudentID_LastName.zip`

**Q: Can I change my feedback questionnaire after submitting?**

A: Of course! Just re-upload - Moodle keeps the latest version.

---

## 💡 This Assignment is Really Simple!

If you find this assignment too easy, it means you're well prepared! Later assignments will gradually increase in difficulty.

If you find it a bit challenging, don't worry:
- This assignment doesn't affect your overall grade (or counts very little)
- The purpose is to help you familiarize with the process
- We'll adjust the course based on everyone's feedback
- Office hours and TA support are available

---

## 📚 Learning Resources

If you want to preview:

**Python Basics**:
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)

**NumPy Introduction**:
- [NumPy Official Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy for Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)

---

## 🎉 After Completion

After submitting, we'll run automated tests and provide feedback. Results are usually available within one week after the deadline.

Don't worry about test failures! If there are issues, we'll tell you how to improve in the feedback.

**Looking forward to your submission!** If you have any questions, feel free to contact me.

---

**Deadline**: Week 1, Friday 23:59
**Office Hours**: TBD
**Email**: haoyu@hku.hk
**Course Website**: https://jasminehao.com/econ6083-slides/
