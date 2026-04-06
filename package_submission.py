
import os
import shutil
import re
import zipfile

def package_submission():
    base_dir = os.getcwd()
    submission_dir = os.path.join(base_dir, "WavKAN_CL_Submission")
    
    # Files to explicitly include
    required_files = [
        "manuscript_complete.tex",
        "references.bib",
        "sn-jnl.cls",
        "sn-mathphys.bst" # Optional but good
    ]

    # 1. Parse TeX for images
    tex_path = "manuscript_complete.tex"
    if not os.path.exists(tex_path):
        print("❌ manuscript_complete.tex not found!")
        return

    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find images
    images = re.findall(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', content)
    
    # 2. Create Clean Directory
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir)
    
    print(f"📦 Packaging submission into '{submission_dir}'...")

    # 3. Copy Main Files
    missing_critical = []
    for params in required_files:
        if os.path.exists(params):
            shutil.copy(params, submission_dir)
            print(f"  ✅ Copied {params}")
        else:
            if params == "sn-jnl.cls":
                missing_critical.append(params)
                print(f"  ⚠️  MISSING {params} (REQUIRED for compilation)")
            else:
                print(f"  ⚠️  Skipping {params} (Not found)")

    # 4. Copy Images
    print("\nProcessing Images:")
    for img in images:
        if os.path.exists(img):
            shutil.copy(img, submission_dir)
            print(f"  ✅ Copied {img}")
        else:
            print(f"  ❌ MISSING IMAGE: {img}")
            missing_critical.append(img)

    # 5. Create Zip
    zip_name = "WavKAN_CL_Submission.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    
    print(f"\n🤐 Created archive: {zip_name}")

    if missing_critical:
        print("\n" + "="*40)
        print("⚠️  WARNING: CRITICAL FILES MISSING")
        print("="*40)
        for f in missing_critical:
            print(f" - {f}")
        print("Please verify these files before submitting!")
    else:
        print("\n✅ Package is COMPLETE and ready for submission!")

if __name__ == "__main__":
    package_submission()
