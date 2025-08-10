#!/usr/bin/env python3
"""
Automatically update TODO.md with current project status.
This script is run by GitHub Actions on PR merge.
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
import re


def get_git_info():
    """Get current git information."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True
        ).strip()
        
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()[:8]
        
        return {"branch": branch, "commit": commit}
    except:
        return {"branch": "unknown", "commit": "unknown"}


def check_tests():
    """Check if tests are passing."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q"],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0
    except:
        return None


def get_code_stats():
    """Get basic code statistics."""
    stats = {
        "python_files": 0,
        "test_files": 0,
        "total_lines": 0
    }
    
    for path in Path("src").rglob("*.py"):
        stats["python_files"] += 1
        with open(path, 'r') as f:
            stats["total_lines"] += len(f.readlines())
    
    for path in Path("tests").rglob("*.py"):
        stats["test_files"] += 1
    
    return stats


def check_dependencies():
    """Check for outdated dependencies."""
    try:
        # This would normally check for outdated packages
        # For now, just return a placeholder
        return []
    except:
        return []


def update_todo_md():
    """Update TODO.md with current status."""
    
    todo_path = Path("TODO.md")
    
    if not todo_path.exists():
        print("TODO.md not found, skipping update")
        return
    
    with open(todo_path, 'r') as f:
        content = f.read()
    
    # Update last updated date
    today = datetime.now().strftime("%Y-%m-%d")
    content = re.sub(
        r"> Last Updated: \d{4}-\d{2}-\d{2}",
        f"> Last Updated: {today}",
        content
    )
    
    # Get git info
    git_info = get_git_info()
    
    # Check if we have any new completed items based on recent commits
    try:
        recent_commits = subprocess.check_output(
            ["git", "log", "--oneline", "-10"],
            text=True
        ).strip().split('\n')
        
        completed_items = []
        for commit in recent_commits:
            commit_lower = commit.lower()
            if "implement" in commit_lower or "add" in commit_lower or "create" in commit_lower:
                # Extract what was implemented
                if "resume" in commit_lower:
                    completed_items.append("Enhanced batch processing with auto-resume")
                if "todo" in commit_lower:
                    completed_items.append("TODO.md automation")
        
        if completed_items:
            # Update recently completed section
            completed_section = f"\n## ✅ Recently Completed\n"
            for item in completed_items[:5]:  # Keep only last 5
                completed_section += f"- [x] {item} ({today})\n"
            
            # Add previous completed items if they exist
            if "## ✅ Recently Completed" in content:
                old_completed = re.search(
                    r"## ✅ Recently Completed\n(.*?)(?=\n##|\Z)",
                    content,
                    re.DOTALL
                ).group(1)
                old_items = [line for line in old_completed.split('\n') if line.strip().startswith('- [x]')][:10]
                for item in old_items:
                    if item not in completed_section:
                        completed_section += item + "\n"
            
            # Replace the section
            if "## ✅ Recently Completed" in content:
                content = re.sub(
                    r"## ✅ Recently Completed\n.*?(?=\n##|\Z)",
                    completed_section.rstrip() + "\n",
                    content,
                    flags=re.DOTALL
                )
    except:
        pass
    
    # Update code statistics
    stats = get_code_stats()
    
    # Check for new risks or blockers based on file analysis
    blockers = []
    risks = []
    
    # Check if artifacts directory is getting too large
    artifacts_path = Path("artifacts")
    if artifacts_path.exists():
        size_mb = sum(f.stat().st_size for f in artifacts_path.rglob('*') if f.is_file()) / (1024 * 1024)
        if size_mb > 1000:
            risks.append(f"Artifacts directory is {size_mb:.1f}MB - consider cleanup")
    
    # Update performance benchmarks if we have new data
    run_state_files = list(Path("artifacts/eval_only/DO_NOT_TRAIN/results").glob(".run_state.json")) if Path("artifacts/eval_only/DO_NOT_TRAIN/results").exists() else []
    if run_state_files:
        try:
            with open(run_state_files[-1], 'r') as f:
                state = json.load(f)
                if state.get("status") == "completed":
                    total = state.get("total_items", 0)
                    started = datetime.fromisoformat(state.get("started_at", ""))
                    completed = datetime.fromisoformat(state.get("completed_at", ""))
                    duration = (completed - started).total_seconds() / 60
                    
                    # Update benchmark table
                    new_row = f"| Batch Score | {total} | {duration:.1f} min | Auto |"
                    if "| Batch Score |" not in content:
                        # Add to performance benchmarks
                        content = re.sub(
                            r"(\| UMAP Prep.*?\|.*?\|.*?\|)",
                            f"\\1\n{new_row}",
                            content
                        )
        except:
            pass
    
    # Check test status
    tests_passing = check_tests()
    if tests_passing is not None:
        status = "✅ Passing" if tests_passing else "❌ Failing"
        # Add or update test status
        if "## 🧪 Test Status" not in content:
            # Add before Known Issues
            content = re.sub(
                r"(## 🐛 Known Issues)",
                f"## 🧪 Test Status\n{status} (Last checked: {today})\n\n\\1",
                content
            )
        else:
            content = re.sub(
                r"## 🧪 Test Status\n.*?(?=\n##|\Z)",
                f"## 🧪 Test Status\n{status} (Last checked: {today})\n",
                content,
                flags=re.DOTALL
            )
    
    # Update statistics
    stats_section = f"\n## 📈 Project Statistics\n"
    stats_section += f"- Python files: {stats['python_files']}\n"
    stats_section += f"- Test files: {stats['test_files']}\n"
    stats_section += f"- Total lines of code: {stats['total_lines']:,}\n"
    stats_section += f"- Last automated update: {today}\n"
    
    if "## 📈 Project Statistics" in content:
        content = re.sub(
            r"## 📈 Project Statistics\n.*?(?=\n##|\Z)",
            stats_section.rstrip() + "\n",
            content,
            flags=re.DOTALL
        )
    else:
        # Add before Notes for Contributors
        content = re.sub(
            r"(## 📝 Notes for Contributors)",
            stats_section + "\n\\1",
            content
        )
    
    # Write updated content
    with open(todo_path, 'w') as f:
        f.write(content)
    
    print(f"✅ TODO.md updated successfully on {today}")
    print(f"   - Python files: {stats['python_files']}")
    print(f"   - Test files: {stats['test_files']}")
    print(f"   - Git commit: {git_info['commit']}")


if __name__ == "__main__":
    update_todo_md()