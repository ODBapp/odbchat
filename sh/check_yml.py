#!/usr/bin/env python3
import sys, yaml, traceback

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_yaml.py <file.yml>")
        sys.exit(1)
    path = sys.argv[1]
    data = open(path, "r", encoding="utf-8").read()
    try:
        yaml.safe_load(data)  # single-doc quick check
        print("[OK] YAML parsed as single document.")
    except yaml.YAMLError as e:
        print("[WARN] single-doc failed, trying multi-doc...")
        try:
            list(yaml.safe_load_all(data))
            print("[OK] YAML parsed as multi-doc.")
        except yaml.YAMLError as e2:
            print("[ERROR] YAML parse failed.\n")
            print(e2)
            # try to extract problem line
            if hasattr(e2, 'problem_mark') and e2.problem_mark is not None:
                mark = e2.problem_mark
                line_no = mark.line + 1
                print(f"\n--> Problem around line {line_no}, column {mark.column+1}\n")
                lines = data.splitlines()
                start = max(0, line_no-6)
                end = min(len(lines), line_no+5)
                for i in range(start, end):
                    prefix = ">>" if (i+1)==line_no else "  "
                    print(f"{prefix} {i+1:4d}: {lines[i]}")
            else:
                traceback.print_exc()

if __name__ == "__main__":
    main()
