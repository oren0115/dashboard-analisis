"""
🚀 Run All Scripts - Menjalankan seluruh pipeline secara berurutan
Script ini menjalankan semua tahap dari data understanding hingga training model
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Jalankan script Python"""
    print("\n" + "="*60)
    print(f"🚀 {description}")
    print("="*60)
    
    if not os.path.exists(script_name):
        print(f"❌ File {script_name} tidak ditemukan!")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True)
        if result.returncode == 0:
            print(f"\n✅ {description} selesai!")
            return True
        else:
            print(f"\n❌ {description} gagal!")
            return False
    except Exception as e:
        print(f"\n❌ Error menjalankan {script_name}: {e}")
        return False

def main():
    """Main function"""
    print("="*60)
    print("🚀 SENTIMENT ANALYSIS - RUN ALL PIPELINE")
    print("="*60)
    print("\nScript ini akan menjalankan seluruh pipeline:")
    print("1. Data Understanding")
    print("2. Data Preprocessing")
    print("3. Model Training")
    print("\nPastikan file 'Sentiment1.csv' dan 'Train1.csv' sudah ada!")
    
    input("\nTekan Enter untuk melanjutkan...")
    
    # Step 1: Data Understanding
    if not run_script('data_understanding.py', 'Data Understanding'):
        print("\n❌ Pipeline dihentikan karena error pada Data Understanding")
        return
    
    # Step 2: Preprocessing
    if not run_script('preprocessing.py', 'Data Preprocessing'):
        print("\n❌ Pipeline dihentikan karena error pada Preprocessing")
        return
    
    # Step 3: Training
    if not run_script('train_model.py', 'Model Training'):
        print("\n❌ Pipeline dihentikan karena error pada Model Training")
        return
    
    print("\n" + "="*60)
    print("✅ SEMUA TAHAP SELESAI!")
    print("="*60)
    print("\n📋 File yang dihasilkan:")
    print("  - dataset_combined.csv")
    print("  - dataset_preprocessed.csv")
    print("  - preprocessor.pkl")
    print("  - best_model.pkl")
    print("  - models/ (folder berisi semua model)")
    print("  - output/ (folder berisi visualisasi)")
    print("\n🚀 Untuk menjalankan API:")
    print("  python app.py")
    print("\n📊 Untuk menjalankan Dashboard:")
    print("  streamlit run dashboard.py")

if __name__ == "__main__":
    main()

