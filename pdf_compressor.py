#!/usr/bin/env python3
"""
World-Class PDF Compressor
A single-file solution for high-quality PDF compression with advanced features.
"""

import os
import sys
import argparse
import tempfile
import shutil
import math
import time
import warnings
from typing import Optional, Tuple, List, Dict
from enum import Enum, auto

try:
    from PIL import Image
    import fitz  # PyMuPDF
    import numpy as np
    from pdf2image import convert_from_path
    import img2pdf
except ImportError as e:
    print(f"Error: Required module not found. {e}")
    print("Please install dependencies with:")
    print("pip install pymupdf Pillow numpy pdf2image img2pdf")
    sys.exit(1)

__version__ = "1.0.0"
__author__ = "Competition Entry"
__license__ = "Proprietary"

class CompressionMode(Enum):
    """Enumeration of available compression modes."""
    HIGH_QUALITY = auto()
    BALANCED = auto()
    MAX_COMPRESSION = auto()
    IMAGE_ONLY = auto()
    TEXT_ONLY = auto()

class PDFCompressor:
    """
    Advanced PDF compressor with multiple optimization strategies.
    Features intelligent analysis, multi-stage compression, and quality preservation.
    """
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="pdf_compress_")
        self.original_size = 0
        self.compressed_size = 0
        self.verbose = False
        self.quality_settings = {
            CompressionMode.HIGH_QUALITY: {
                'dpi': 300,
                'image_quality': 90,
                'compress_text': True,
                'compress_images': True,
                'downsample_images': False,
                'optimize': 3
            },
            CompressionMode.BALANCED: {
                'dpi': 200,
                'image_quality': 80,
                'compress_text': True,
                'compress_images': True,
                'downsample_images': True,
                'optimize': 2
            },
            CompressionMode.MAX_COMPRESSION: {
                'dpi': 150,
                'image_quality': 70,
                'compress_text': True,
                'compress_images': True,
                'downsample_images': True,
                'optimize': 1
            },
            CompressionMode.IMAGE_ONLY: {
                'dpi': 200,
                'image_quality': 85,
                'compress_text': False,
                'compress_images': True,
                'downsample_images': True,
                'optimize': 2
            },
            CompressionMode.TEXT_ONLY: {
                'dpi': 300,
                'image_quality': 100,
                'compress_text': True,
                'compress_images': False,
                'downsample_images': False,
                'optimize': 3
            }
        }
    
    def __del__(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def log(self, message: str):
        """Log messages when verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def analyze_pdf(self, input_path: str) -> Dict:
        """
        Analyze PDF structure and content.
        Returns a dictionary with analysis results.
        """
        analysis = {
            'pages': 0,
            'text_pages': 0,
            'image_pages': 0,
            'mixed_pages': 0,
            'total_images': 0,
            'image_types': {},
            'fonts': set(),
            'metadata': {}
        }
        
        try:
            doc = fitz.open(input_path)
            analysis['pages'] = len(doc)
            analysis['metadata'] = doc.metadata
            
            for page in doc:
                has_text = False
                has_images = False
                
                # Check for text
                text = page.get_text()
                if text.strip():
                    has_text = True
                    analysis['text_pages'] += 1
                
                # Check for images
                img_list = page.get_images(full=True)
                if img_list:
                    has_images = True
                    analysis['total_images'] += len(img_list)
                    for img in img_list:
                        xref = img[0]
                        img_info = doc.extract_image(xref)
                        img_type = img_info['ext']
                        analysis['image_types'][img_type] = analysis['image_types'].get(img_type, 0) + 1
                
                if has_text and has_images:
                    analysis['mixed_pages'] += 1
                elif has_images:
                    analysis['image_pages'] += 1
                
                # Extract fonts
                for block in page.get_text("dict")["blocks"]:
                    if 'lines' in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                analysis['fonts'].add(span["font"])
            
            doc.close()
        except Exception as e:
            warnings.warn(f"Analysis failed: {str(e)}")
        
        return analysis
    
    def optimize_pdf(self, input_path: str, output_path: str, mode: CompressionMode = CompressionMode.BALANCED,
                    custom_dpi: Optional[int] = None, custom_quality: Optional[int] = None) -> Tuple[bool, str]:
        """
        Optimize and compress PDF file with advanced techniques.
        Returns (success, message) tuple.
        """
        if not os.path.isfile(input_path):
            return False, "Input file does not exist"
        
        self.original_size = os.path.getsize(input_path)
        if self.original_size == 0:
            return False, "Input file is empty"
        
        settings = self.quality_settings[mode]
        if custom_dpi:
            settings['dpi'] = custom_dpi
        if custom_quality:
            settings['image_quality'] = custom_quality
        
        analysis = self.analyze_pdf(input_path)
        self.log(f"PDF Analysis: {analysis}")
        
        try:
            # Step 1: Initial optimization with PyMuPDF
            temp_pdf1 = os.path.join(self.temp_dir, "stage1.pdf")
            self._optimize_with_pymupdf(input_path, temp_pdf1, settings)
            
            # Step 2: Image optimization if needed
            if settings['compress_images'] and analysis['total_images'] > 0:
                temp_pdf2 = os.path.join(self.temp_dir, "stage2.pdf")
                self._optimize_images(temp_pdf1, temp_pdf2, settings)
                intermediate_path = temp_pdf2
            else:
                intermediate_path = temp_pdf1
            
            # Step 3: Final optimization pass
            self._final_optimization(intermediate_path, output_path, settings)
            
            self.compressed_size = os.path.getsize(output_path)
            compression_ratio = (1 - self.compressed_size / self.original_size) * 100
            
            result_msg = (f"Compression successful. Original: {self._format_bytes(self.original_size)}, "
                         f"Compressed: {self._format_bytes(self.compressed_size)}, "
                         f"Reduction: {compression_ratio:.1f}%")
            
            return True, result_msg
        except Exception as e:
            return False, f"Compression failed: {str(e)}"
    
    def _optimize_with_pymupdf(self, input_path: str, output_path: str, settings: Dict):
        """First stage optimization using PyMuPDF's built-in features."""
        doc = fitz.open(input_path)
        
        # Enable compression for all possible elements
        for page in doc:
            # Clean and sanitize the page
            page.clean_contents()
            
            if settings['compress_text']:
                # Attempt to consolidate text objects
                page.wrap_contents()
        
        # Save with optimization options
        save_options = {
            'garbage': 4,  # aggressive garbage collection
            'deflate': True,  # compress streams
            'linear': False,  # don't linearize (faster)
            'clean': True,  # clean and sanitize content
            'pretty': False,  # don't pretty-print (smaller)
            'compress': True,  # compress everything
            'compress_images': settings['compress_images'],
            'compress_fonts': settings['compress_text']
        }
        
        doc.save(output_path, **save_options)
        doc.close()
    
    def _optimize_images(self, input_path: str, output_path: str, settings: Dict):
        """
        Advanced image optimization pipeline.
        Converts PDF to images, optimizes them, and recreates the PDF.
        """
        # Convert PDF to images
        images = convert_from_path(
            input_path,
            dpi=settings['dpi'],
            fmt='jpeg' if settings['image_quality'] < 95 else 'png',
            thread_count=4,
            output_folder=self.temp_dir,
            paths_only=True
        )
        
        optimized_images = []
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    # Convert to efficient mode if needed
                    if img.mode not in ['L', 'RGB', 'CMYK']:
                        img = img.convert('RGB')
                    
                    # Downsample if requested
                    if settings['downsample_images']:
                        current_dpi = img.info.get('dpi', (settings['dpi'], settings['dpi']))
                        if min(current_dpi) > settings['dpi']:
                            scale_factor = settings['dpi'] / min(current_dpi)
                            new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Optimize and save
                    optimized_path = os.path.join(self.temp_dir, f"opt_{os.path.basename(img_path)}")
                    if img_path.lower().endswith('.png'):
                        img.save(optimized_path, 'PNG', optimize=True, compress_level=settings['optimize'])
                    else:
                        img.save(optimized_path, 'JPEG', quality=settings['image_quality'], optimize=True, 
                                progressive=True, subsampling='4:2:0' if settings['image_quality'] < 90 else '4:4:4')
                    
                    optimized_images.append(optimized_path)
            except Exception as e:
                warnings.warn(f"Failed to optimize image {img_path}: {str(e)}")
                optimized_images.append(img_path)  # fallback to original
        
        # Recreate PDF from optimized images
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(optimized_images, layout_fun=img2pdf.get_layout_fun(
                fitz.open(input_path).load_page(0).rect.width,
                fitz.open(input_path).load_page(0).rect.height
            )))
        
        # Clean up temporary image files
        for img_path in images + optimized_images:
            try:
                os.remove(img_path)
            except:
                pass
    
    def _final_optimization(self, input_path: str, output_path: str, settings: Dict):
        """Final optimization pass to clean up and compress further."""
        doc = fitz.open(input_path)
        
        # Additional optimization options
        save_options = {
            'garbage': 4,
            'deflate': True,
            'clean': True,
            'compress': True,
            'compress_images': False,  # already compressed
            'compress_fonts': settings['compress_text'],
            'pretty': False
        }
        
        doc.save(output_path, **save_options)
        doc.close()
    
    def _format_bytes(self, size: int) -> str:
        """Format file size in human-readable format."""
        if size == 0:
            return "0B"
        
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size, 1024)))
        p = math.pow(1024, i)
        s = round(size / p, 2)
        return f"{s} {size_name[i]}"
    
    def set_verbose(self, verbose: bool):
        """Enable or disable verbose logging."""
        self.verbose = verbose

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced PDF Compressor - World-Class Optimization Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input', help="Input PDF file path")
    parser.add_argument('output', help="Output PDF file path")
    parser.add_argument('-m', '--mode', choices=['high', 'balanced', 'max', 'image', 'text'],
                       default='balanced', help="Compression mode")
    parser.add_argument('--dpi', type=int, help="Custom DPI for image downsampling")
    parser.add_argument('--quality', type=int, choices=range(1, 101), metavar="1-100",
                       help="Custom image quality (1-100)")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--version', action='version', version=f"%(prog)s {__version__}")
    
    return parser.parse_args()

def main():
    """Main entry point for command line execution."""
    args = parse_args()
    
    # Map mode argument to enum
    mode_mapping = {
        'high': CompressionMode.HIGH_QUALITY,
        'balanced': CompressionMode.BALANCED,
        'max': CompressionMode.MAX_COMPRESSION,
        'image': CompressionMode.IMAGE_ONLY,
        'text': CompressionMode.TEXT_ONLY
    }
    
    compressor = PDFCompressor()
    compressor.set_verbose(args.verbose)
    
    success, message = compressor.optimize_pdf(
        input_path=args.input,
        output_path=args.output,
        mode=mode_mapping[args.mode],
        custom_dpi=args.dpi,
        custom_quality=args.quality
    )
    
    if success:
        print(message)
        print(f"Output saved to: {os.path.abspath(args.output)}")
    else:
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Disable PIL decompression bomb warning
    Image.MAX_IMAGE_PIXELS = None
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)
    
    # Run the application
    main()
