# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import base64
import logging
from PIL import Image
import io
import asyncio

# Claude's maximum image size limit (5MB)
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB

def get_file_size(file_path):
    """Get file size in bytes."""
    return os.path.getsize(file_path)

def get_base64_size(base64_string):
    """Get the size of a base64 encoded string in bytes."""
    return len(base64_string.encode('utf-8'))

def compress_image_to_limit(image_path, max_size_bytes=MAX_IMAGE_SIZE_BYTES, quality_start=85, min_quality=20):
    """
    Compress an image to stay under the specified size limit.
    
    Args:
        image_path (str): Path to the input image
        max_size_bytes (int): Maximum allowed size in bytes (default: 5MB)
        quality_start (int): Starting JPEG quality (default: 85)
        min_quality (int): Minimum JPEG quality to try (default: 20)
    
    Returns:
        str: Base64 encoded compressed image
        
    Raises:
        ValueError: If image cannot be compressed to fit within size limit
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # First check if the original file is already under the limit
    original_size = get_file_size(image_path)
    logger.info(f"Original image size: {original_size / (1024*1024):.2f} MB")
    
    # Open and convert image
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (for JPEG compatibility)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create a white background for transparent images
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Try different compression levels
        quality = quality_start
        while quality >= min_quality:
            # Compress image to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=quality, optimize=True)
            img_bytes = img_buffer.getvalue()
            
            # Encode to base64 and check size
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            base64_size = get_base64_size(base64_string)
            
            logger.info(f"Quality {quality}: {base64_size / (1024*1024):.2f} MB")
            
            if base64_size <= max_size_bytes:
                logger.info(f"Successfully compressed image to {base64_size / (1024*1024):.2f} MB at quality {quality}")
                return base64_string
            
            # Reduce quality for next iteration
            quality -= 10
        
        # If still too large, try reducing image dimensions
        logger.warning("Quality reduction insufficient, trying dimension reduction")
        
        # Try reducing dimensions by 10% each iteration
        scale_factor = 0.9
        current_img = img.copy()
        
        while scale_factor >= 0.3:  # Don't go below 30% of original size
            new_width = int(current_img.width * scale_factor)
            new_height = int(current_img.height * scale_factor)
            resized_img = current_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Try with moderate quality
            img_buffer = io.BytesIO()
            resized_img.save(img_buffer, format='JPEG', quality=70, optimize=True)
            img_bytes = img_buffer.getvalue()
            
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            base64_size = get_base64_size(base64_string)
            
            logger.info(f"Resized to {new_width}x{new_height} (scale {scale_factor:.1f}): {base64_size / (1024*1024):.2f} MB")
            
            if base64_size <= max_size_bytes:
                logger.info(f"Successfully compressed image to {base64_size / (1024*1024):.2f} MB with {scale_factor:.1f}x scale")
                return base64_string
            
            scale_factor -= 0.1
        
        # If we still can't fit it, raise an error
        raise ValueError(f"Unable to compress image to fit within {max_size_bytes / (1024*1024):.1f} MB limit")

def encode_image_with_compression(image_path, max_size_bytes=MAX_IMAGE_SIZE_BYTES):
    """
    Encode an image to base64 with automatic compression if needed.
    
    Args:
        image_path (str): Path to the image file
        max_size_bytes (int): Maximum allowed size in bytes (default: 5MB)
    
    Returns:
        str: Base64 encoded image (compressed if necessary)
    """
    logger = logging.getLogger(__name__)
    
    if image_path is None:
        raise ValueError("Image path cannot be None")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # First try the original file
    try:
        with open(image_path, "rb") as image_file:
            original_bytes = image_file.read()
            base64_string = base64.b64encode(original_bytes).decode('utf-8')
            base64_size = get_base64_size(base64_string)
            
            if base64_size <= max_size_bytes:
                logger.info(f"Original image fits within limit: {base64_size / (1024*1024):.2f} MB")
                return base64_string
            else:
                logger.info(f"Original image too large: {base64_size / (1024*1024):.2f} MB, compressing...")
                return compress_image_to_limit(image_path, max_size_bytes)
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        # Fallback to compression
        return compress_image_to_limit(image_path, max_size_bytes)


async def take_screenshot(agent):
    try:
        os.makedirs(os.path.join(agent.main_path, 'screenshots'), exist_ok=True)
        screenshot_path = os.path.join(agent.main_path, 'screenshots', f'screen_{agent.time_step}.png')
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            try:
                try:
                    await agent.page.wait_for_load_state('domcontentloaded', timeout=15000)
                except Exception:
                    pass
                try:
                    await agent.page.wait_for_load_state('networkidle', timeout=20000)
                except Exception:
                    pass
                await agent.page.screenshot(path=screenshot_path, timeout=45000)
                agent.screenshot_path = screenshot_path
                try:
                    with Image.open(agent.screenshot_path) as img:
                        extrema = img.getextrema()
                        uniform = True
                        for band in extrema:
                            if isinstance(band, tuple) and band[0] != band[1]:
                                uniform = False
                                break
                        if uniform:
                            try:
                                lum = img.convert("L")
                                hist = lum.histogram()
                                total = sum(hist) or 1
                                white_ratio = hist[-1] / float(total)
                                if white_ratio < 0.99:
                                    uniform = False
                            except Exception:
                                pass
                        if not uniform:
                            break
                except Exception:
                    break
                attempts += 1
                try:
                    await agent.page.evaluate("window.scrollTo(0, 0)")
                except Exception:
                    pass
                try:
                    await asyncio.sleep(1)
                except Exception:
                    pass
                if attempts == 2:
                    try:
                        context = agent.session_control.get('context') if isinstance(agent.session_control, dict) else None
                        target_url = None
                        if hasattr(agent, 'page') and agent.page:
                            try:
                                target_url = agent.page.url
                            except Exception:
                                target_url = None
                        if context:
                            new_page = await context.new_page()
                            await agent.page_on_open_handler(new_page)
                            if target_url:
                                try:
                                    await new_page.goto(target_url, wait_until="domcontentloaded")
                                except Exception:
                                    pass
                            agent.page = new_page
                    except Exception:
                        pass
            except Exception:
                break
        agent.logger.info(f"Viewport screenshot taken: {agent.screenshot_path}")
        try:
            if agent.config.get('agent', {}).get('highlight', False):
                await agent._annotate_current_screenshot()
        except Exception as _ann_e:
            agent.logger.warning(f"Failed to annotate screenshot: {_ann_e}")
    except Exception as e:
        agent.logger.warning(f"Failed to take screenshot: {e}")
        try:
            try:
                await agent.page.wait_for_load_state('domcontentloaded', timeout=10000)
            except Exception:
                pass
            try:
                await agent.page.screenshot(path=screenshot_path, timeout=45000)
                agent.screenshot_path = screenshot_path
                agent.logger.info(f"Viewport screenshot taken: {agent.screenshot_path}")
                return
            except Exception:
                pass
            context = agent.session_control.get('context') if isinstance(agent.session_control, dict) else None
            if context:
                new_page = await context.new_page()
                await agent.page_on_open_handler(new_page)
                target_url = None
                last_resp = agent.session_control.get('last_response') if isinstance(agent.session_control, dict) else None
                if last_resp and isinstance(last_resp, dict):
                    target_url = last_resp.get('url')
                if not target_url and hasattr(agent, 'actual_website'):
                    target_url = agent.actual_website
                if not target_url:
                    try:
                        pages = context.pages
                        if pages:
                            target_url = pages[-1].url
                    except Exception:
                        target_url = None
                if target_url:
                    try:
                        await new_page.goto(target_url, wait_until="domcontentloaded")
                    except Exception:
                        pass
                agent.page = new_page
                if target_url:
                    await agent.page.screenshot(path=screenshot_path, timeout=45000)
                    agent.screenshot_path = screenshot_path
                    agent.logger.info(f"Viewport screenshot taken: {agent.screenshot_path}")
                    return
        except Exception as rec_e:
            agent.logger.error(f"Screenshot capture failed: {rec_e}")
        agent.screenshot_path = None


async def annotate_current_screenshot(agent):
    try:
        if not agent.screenshot_path or not os.path.exists(agent.screenshot_path):
            return
        try:
            timg = Image.open(agent.screenshot_path)
            tex = timg.getextrema()
            tall_same = True
            for band in tex:
                if isinstance(band, tuple) and band[0] != band[1]:
                    tall_same = False
                    break
            if tall_same:
                return
        except Exception:
            pass
        active = await agent.page.evaluate(
            """
            () => {
                const ae = document.activeElement;
                if (ae && ae !== document.body) {
                    const r = ae.getBoundingClientRect();
                    return {
                        tag: ae.tagName,
                        placeholder: ae.placeholder || '',
                        id: ae.id || '',
                        rect: { left: r.left, top: r.top, width: r.width, height: r.height }
                    };
                }
                return null;
            }
            """
        )
        from PIL import ImageDraw, ImageFont
        img = Image.open(agent.screenshot_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(img)
        vs = getattr(agent.page, 'viewport_size', None)
        if isinstance(vs, dict) and 'width' in vs and 'height' in vs and vs['width'] > 0 and vs['height'] > 0:
            sx = img.width / float(vs['width'])
            sy = img.height / float(vs['height'])
        else:
            vx = agent.config.get('browser', {}).get('viewport', {}).get('width', img.width)
            vy = agent.config.get('browser', {}).get('viewport', {}).get('height', img.height)
            sx = img.width / float(vx) if vx else 1.0
            sy = img.height / float(vy) if vy else 1.0
        if active and isinstance(active, dict):
            rect = active.get('rect') or {}
            w = rect.get('width') or 0
            h = rect.get('height') or 0
            if w > 0 and h > 0:
                left = rect.get('left') or 0
                top = rect.get('top') or 0
                x0 = int(max(0, left * sx))
                y0 = int(max(0, top * sy))
                x1 = int(min(img.width - 1, (left + w) * sx))
                y1 = int(min(img.height - 1, (top + h) * sy))
                x1 = max(x0, x1)
                y1 = max(y0, y1)
                draw.rectangle([(x0, y0), (x1, y1)], outline='lime', width=3)
                label_text = (active.get('tag') or '') + ' ' + (active.get('placeholder') or active.get('id') or '')
                if label_text:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                    except Exception:
                        font = ImageFont.load_default()
                    tb = draw.textbbox((0, 0), label_text, font=font)
                    tw = tb[2] - tb[0]
                    th = tb[3] - tb[1]
                    tx = max(0, min(x0, img.width - max(6, tw) - 6))
                    ty = max(0, min(max(0, y0 - th - 6), img.height - max(6, th) - 6))
                    draw.rectangle([(tx - 3, ty - 3), (tx + tw + 3, ty + th + 3)], fill='black')
                    draw.text((tx, ty), label_text, fill='lime', font=font)
        if getattr(agent, 'last_click_viewport_coords', None):
            cx, cy = agent.last_click_viewport_coords
            cx = int(round(cx * sx))
            cy = int(round(cy * sy))
            ms = 16
            lw = 3
            ex0 = max(0, cx - 30)
            ey0 = max(0, cy - 30)
            ex1 = min(img.width - 1, cx + 30)
            ey1 = min(img.height - 1, cy + 30)
            ex1 = max(ex0 + 1, ex1)
            ey1 = max(ey0 + 1, ey1)
            odraw.ellipse([(ex0, ey0), (ex1, ey1)], fill=(0, 255, 0, 96))
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)
            draw.line([(cx, cy - ms), (cx, cy + ms)], fill='red', width=lw)
            draw.line([(cx - ms, cy), (cx + ms, cy)], fill='red', width=lw)
            dx0 = max(0, cx - 8)
            dy0 = max(0, cy - 8)
            dx1 = min(img.width - 1, cx + 8)
            dy1 = min(img.height - 1, cy + 8)
            dx1 = max(dx0 + 1, dx1)
            dy1 = max(dy0 + 1, dy1)
            draw.ellipse([(dx0, dy0), (dx1, dy1)], outline='red', width=lw)
        labeled_path = os.path.join(agent.main_path, 'screenshots', f'screen_{agent.time_step}_labeled.png')
        img.save(labeled_path)
        agent.logger.info(f"Labeled screenshot saved: {labeled_path}")
    except Exception as e:
        try:
            agent.logger.warning(f"Failed to save labeled screenshot: {e}")
        except Exception:
            pass


async def take_full_page_screenshot_with_cropping(agent, target_elements=None, screenshot_path=None):
    if screenshot_path is None:
        screenshot_path = agent.screenshot_path
        
    try:
        total_height = await agent.page.evaluate('''() => {
            return Math.max(
                document.documentElement.scrollHeight, 
                document.body.scrollHeight,
                document.documentElement.clientHeight
            );
        }''')
        
        viewport_size = agent.page.viewport_size
        if viewport_size is None:
            viewport_size = {'width': 1280, 'height': 720}
        total_width = viewport_size['width']
        
        if target_elements and len(target_elements) > 0:
            try:
                element_positions = []
                for element_info in target_elements:
                    if isinstance(element_info, dict) and 'y' in element_info:
                        element_positions.append(element_info['y'])
                    elif hasattr(element_info, 'y'):
                        element_positions.append(element_info.y)
                
                if element_positions:
                    height_start = min(element_positions)
                    height_end = max(element_positions)
                    
                    clip_start = min(total_height - 1144, max(0, height_start - 200))
                    clip_height = min(total_height - clip_start, max(height_end - height_start + 200, 1144))
                    clip = {"x": 0, "y": clip_start, "width": total_width, "height": clip_height}
                    
                    await agent.page.screenshot(
                        path=screenshot_path, 
                        clip=clip, 
                        full_page=True,
                        type='png',
                        timeout=20000
                    )
                    agent.logger.info(f"Smart cropped screenshot taken with clip: {clip}")
                    return
            except Exception as crop_error:
                agent.logger.warning(f"Smart cropping failed: {crop_error}, falling back to full page")
        
        await agent.page.screenshot(
            path=screenshot_path, 
            full_page=True,
            type='png',
            timeout=20000
        )
        agent.logger.info("Full page screenshot taken as fallback")
        
    except Exception as e:
        await agent.page.screenshot(path=screenshot_path)
        agent.logger.warning(f"Full page screenshot failed, took viewport screenshot: {e}")
