"""
Apple Silicon Compatibility Module

This module provides compatibility utilities for running the robot arm simulation
on Apple Silicon Macs, particularly addressing tkinter compatibility issues.
"""

import platform
import sys
import tkinter as tk
from typing import Dict, Any, Optional


class AppleSiliconCompat:
    """Apple Silicon compatibility utilities."""
    
    def __init__(self):
        """Initialize compatibility checker."""
        self.is_apple_silicon = self._check_apple_silicon()
        self.is_macos = platform.system() == "Darwin"
        self.macos_version = self._get_macos_version()
        self.tkinter_version = self._get_tkinter_version()
        
        # Compatibility flags
        self.supports_bg_option = self._check_bg_option_support()
        self.supports_fg_option = self._check_fg_option_support()
        
        if self.is_apple_silicon:
            print(f"🍎 Apple Silicon detected (macOS {self.macos_version})")
            print(f"   tkinter version: {self.tkinter_version}")
            print(f"   Background color support: {'✅' if self.supports_bg_option else '❌'}")
            print(f"   Foreground color support: {'✅' if self.supports_fg_option else '❌'}")
    
    def _check_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        try:
            import subprocess
            result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
            return result.stdout.strip() == "arm64"
        except:
            return False
    
    def _get_macos_version(self) -> str:
        """Get macOS version."""
        if not self.is_macos:
            return "N/A"
        
        try:
            import subprocess
            result = subprocess.run(['sw_vers', '-productVersion'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "Unknown"
    
    def _get_tkinter_version(self) -> str:
        """Get tkinter version."""
        try:
            return tk.TkVersion
        except:
            return "Unknown"
    
    def _check_bg_option_support(self) -> bool:
        """Check if tkinter supports the -bg option."""
        if not self.is_apple_silicon:
            return True  # Assume support on non-Apple Silicon
        
        try:
            # Create a temporary root window to test
            root = tk.Tk()
            root.withdraw()  # Hide the window
            
            # Try to create a label with bg option
            test_label = tk.Label(root, text="test")
            test_label.config(bg="white")
            
            # Clean up
            root.destroy()
            return True
            
        except tk.TclError:
            return False
        except Exception:
            return False
    
    def _check_fg_option_support(self) -> bool:
        """Check if tkinter supports the -fg option."""
        if not self.is_apple_silicon:
            return True  # Assume support on non-Apple Silicon
        
        try:
            # Create a temporary root window to test
            root = tk.Tk()
            root.withdraw()  # Hide the window
            
            # Try to create a label with fg option
            test_label = tk.Label(root, text="test")
            test_label.config(fg="black")
            
            # Clean up
            root.destroy()
            return True
            
        except tk.TclError:
            return False
        except Exception:
            return False
    
    def safe_config_bg(self, widget, color: str) -> bool:
        """Safely configure background color for a widget.
        
        Args:
            widget: tkinter widget to configure
            color: color name or hex value
            
        Returns:
            True if successful, False if not supported
        """
        if not self.supports_bg_option:
            return False
        
        try:
            widget.config(bg=color)
            return True
        except tk.TclError:
            # Try alternative method
            try:
                widget.config(background=color)
                return True
            except tk.TclError:
                return False
        except Exception:
            return False
    
    def safe_config_fg(self, widget, color: str) -> bool:
        """Safely configure foreground color for a widget.
        
        Args:
            widget: tkinter widget to configure
            color: color name or hex value
            
        Returns:
            True if successful, False if not supported
        """
        if not self.supports_fg_option:
            return False
        
        try:
            widget.config(fg=color)
            return True
        except tk.TclError:
            # Try alternative method
            try:
                widget.config(foreground=color)
                return True
            except tk.TclError:
                return False
        except Exception:
            return False
    
    def safe_config_colors(self, widget, bg: Optional[str] = None, 
                          fg: Optional[str] = None) -> Dict[str, bool]:
        """Safely configure both background and foreground colors.
        
        Args:
            widget: tkinter widget to configure
            bg: background color (optional)
            fg: foreground color (optional)
            
        Returns:
            Dictionary with success status for each color
        """
        results = {}
        
        if bg is not None:
            results['bg'] = self.safe_config_bg(widget, bg)
        
        if fg is not None:
            results['fg'] = self.safe_config_fg(widget, fg)
        
        return results
    
    def get_alternative_visual_indicator(self, status: str) -> Dict[str, str]:
        """Get alternative visual indicators when colors are not supported.
        
        Args:
            status: status type (e.g., 'ok', 'warning', 'error')
            
        Returns:
            Dictionary with text and symbol alternatives
        """
        indicators = {
            'ok': {'symbol': '✅', 'text': '[OK]', 'prefix': '✓ '},
            'good': {'symbol': '✅', 'text': '[GOOD]', 'prefix': '✓ '},
            'warning': {'symbol': '⚠️', 'text': '[WARN]', 'prefix': '⚠ '},
            'error': {'symbol': '❌', 'text': '[ERROR]', 'prefix': '✗ '},
            'limit': {'symbol': '🚫', 'text': '[LIMIT]', 'prefix': '! '},
            'moving': {'symbol': '🔄', 'text': '[MOVING]', 'prefix': '→ '},
            'enabled': {'symbol': '🟢', 'text': '[ENABLED]', 'prefix': '● '},
            'disabled': {'symbol': '🔴', 'text': '[DISABLED]', 'prefix': '○ '}
        }
        
        return indicators.get(status, {'symbol': '●', 'text': f'[{status.upper()}]', 'prefix': '• '})
    
    def create_status_indicator(self, parent, status: str, text: str = "") -> tk.Label:
        """Create a status indicator that works on Apple Silicon.
        
        Args:
            parent: parent widget
            status: status type
            text: additional text to display
            
        Returns:
            Configured label widget
        """
        indicator_info = self.get_alternative_visual_indicator(status)
        
        if self.supports_fg_option:
            # Use colors if supported
            color_map = {
                'ok': 'green',
                'good': 'green', 
                'warning': 'orange',
                'error': 'red',
                'limit': 'red',
                'moving': 'blue',
                'enabled': 'green',
                'disabled': 'red'
            }
            
            color = color_map.get(status, 'black')
            label_text = f"{indicator_info['prefix']}{text}" if text else indicator_info['symbol']
            
            label = tk.Label(parent, text=label_text, fg=color, font=("Arial", 10))
            
        else:
            # Use symbols and text when colors not supported
            label_text = f"{indicator_info['symbol']} {text}" if text else indicator_info['symbol']
            label = tk.Label(parent, text=label_text, font=("Arial", 10))
        
        return label
    
    def print_compatibility_info(self):
        """Print detailed compatibility information."""
        print("\n" + "="*60)
        print("🍎 APPLE SILICON COMPATIBILITY REPORT")
        print("="*60)
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.machine()}")
        print(f"Python: {sys.version}")
        
        if self.is_macos:
            print(f"macOS Version: {self.macos_version}")
        
        print(f"tkinter Version: {self.tkinter_version}")
        print(f"Apple Silicon: {'Yes' if self.is_apple_silicon else 'No'}")
        
        print("\nGUI Compatibility:")
        print(f"  Background colors (-bg): {'✅ Supported' if self.supports_bg_option else '❌ Not supported'}")
        print(f"  Foreground colors (-fg): {'✅ Supported' if self.supports_fg_option else '❌ Not supported'}")
        
        if not self.supports_bg_option or not self.supports_fg_option:
            print("\n💡 Workarounds enabled:")
            print("  • Using symbols and text for status indicators")
            print("  • Alternative visual feedback methods")
            print("  • Graceful degradation for unsupported features")
        
        print("="*60)


# Global compatibility instance
_compat_instance = None

def get_compat() -> AppleSiliconCompat:
    """Get the global compatibility instance."""
    global _compat_instance
    if _compat_instance is None:
        _compat_instance = AppleSiliconCompat()
    return _compat_instance


def safe_config_widget_colors(widget, bg: Optional[str] = None, 
                             fg: Optional[str] = None) -> Dict[str, bool]:
    """Convenience function to safely configure widget colors."""
    return get_compat().safe_config_colors(widget, bg, fg)


def create_status_indicator(parent, status: str, text: str = "") -> tk.Label:
    """Convenience function to create status indicators."""
    return get_compat().create_status_indicator(parent, status, text)


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return get_compat().is_apple_silicon


def supports_colors() -> bool:
    """Check if the system supports tkinter color options."""
    compat = get_compat()
    return compat.supports_bg_option and compat.supports_fg_option
