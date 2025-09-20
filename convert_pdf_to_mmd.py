#!/usr/bin/env python3
"""
Batch PDF to MMD Converter using Marker
- Input: database/ folder with PDF files
- Output: files_mmd/ folder with MMD files only
- Features: 8 concurrent processes, 10-minute timeout per file
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import signal

# Configuration
DATABASE_DIR = Path("database")
OUTPUT_DIR = Path("files_mmd")
MAX_WORKERS = 8
TIMEOUT_SECONDS = None  # No timeout limit
LOG_FILE = "marker_conversion.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

class PDFConverter:
    def __init__(self):
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        DATABASE_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        logging.info(f" Input directory: {DATABASE_DIR.absolute()}")
        logging.info(f" Output directory: {OUTPUT_DIR.absolute()}")
    
    def get_pdf_files(self):
        """Get all PDF files"""
        pdf_files = list(DATABASE_DIR.glob("*.pdf"))
        logging.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    def convert_single_pdf(self, pdf_path, progress_dict, pdf_index, total_files):
        """Convert single PDF file"""
        try:
            start_time = time.time()
            pdf_name = pdf_path.stem
            
            # Create temporary output directory (marker will create subfolders)
            temp_output = OUTPUT_DIR / "temp_marker_output"
            temp_output.mkdir(exist_ok=True)
            
            logging.info(f"[{pdf_index}/{total_files}] Start conversion: {pdf_name}")
            
            # Build marker command
            cmd = [
                "marker_single",
                str(pdf_path),
                "--output_dir", str(temp_output),
                "--output_format", "markdown",
                "--force_ocr",
                "--disable_image_extraction"
            ]
            
            # Execute conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Find generated markdown files
                marker_output_dir = temp_output / pdf_name
                md_files = list(marker_output_dir.glob("*.md"))
                
                if md_files:
                    # Move and rename to .mmd
                    source_md = md_files[0]
                    target_mmd = OUTPUT_DIR / f"{pdf_name}.mmd"
                    
                    # Copy content to .mmd file
                    with open(source_md, 'r', encoding='utf-8') as src:
                        content = src.read()
                    
                    with open(target_mmd, 'w', encoding='utf-8') as dst:
                        dst.write(content)
                    
                    # Clean up temporary files
                    import shutil
                    shutil.rmtree(marker_output_dir, ignore_errors=True)
                    
                    elapsed = time.time() - start_time
                    progress_dict[pdf_name] = "success"
                    logging.info(f"[{pdf_index}/{total_files}] Completed: {pdf_name} ({elapsed:.1f}s)")
                    return {"status": "success", "file": pdf_name, "time": elapsed}
                else:
                    progress_dict[pdf_name] = "no_output"
                    logging.error(f"[{pdf_index}/{total_files}] No output found: {pdf_name}")
                    return {"status": "no_output", "file": pdf_name}
            else:
                progress_dict[pdf_name] = "failed"
                logging.error(f"[{pdf_index}/{total_files}] Conversion failed: {pdf_name}")
                logging.error(f"Error output: {result.stderr}")
                return {"status": "failed", "file": pdf_name, "error": result.stderr}
                
        except Exception as e:
            progress_dict[pdf_name] = "error"
            logging.error(f" [{pdf_index}/{total_files}] Exception error: {pdf_name} - {str(e)}")
            return {"status": "error", "file": pdf_name, "error": str(e)}
    
    def batch_convert(self):
        """Batch convert PDF files"""
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            logging.warning("No PDF files found")
            return
        
        # Create progress tracking
        manager = Manager()
        progress_dict = manager.dict()
        
        results = {"success": 0, "failed": 0, "error": 0, "no_output": 0}
        
        logging.info(f"Start batch conversion of {len(pdf_files)} files, using {MAX_WORKERS} concurrent processes")
        
        start_time = time.time()
        
        # Use process pool for concurrent processing
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(
                    self.convert_single_pdf, 
                    pdf_path, 
                    progress_dict, 
                    idx + 1, 
                    len(pdf_files)
                ): pdf_path 
                for idx, pdf_path in enumerate(pdf_files)
            }
            
            # Collect results
            for future in as_completed(future_to_pdf):
                result = future.result()
                results[result["status"]] += 1
                
                # Show progress
                completed = sum(results.values())
                logging.info(f"Progress: {completed}/{len(pdf_files)} "
                           f"({results['success']} {results['failed']} "
                           f"{results['error']} {results['no_output']})")
        
        # Summary
        total_time = time.time() - start_time
        logging.info(f"\n Batch conversion completed!")
        logging.info(f"Total time: {total_time:.1f} seconds")
        logging.info(f"Success: {results['success']}")
        logging.info(f"Failed: {results['failed']}")
        logging.info(f"Error: {results['error']}")
        logging.info(f"No output: {results['no_output']}")
        
        # List generated files
        mmd_files = list(OUTPUT_DIR.glob("*.mmd"))
        logging.info(f" Generated MMD files: {len(mmd_files)}")
        for mmd_file in mmd_files[:5]:  # Only show first 5
            logging.info(f"   📄 {mmd_file.name}")
        if len(mmd_files) > 5:
            logging.info(f"   ... and {len(mmd_files) - 5} other files")

def main():
    """Main function"""
    logging.info(" Starting Marker PDF to MMD batch converter")
    
    converter = PDFConverter()
    
    try:
        converter.batch_convert()
    except KeyboardInterrupt:
        logging.info("User interrupted conversion")
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
    
    logging.info("Program ended")

if __name__ == "__main__":
    main()
