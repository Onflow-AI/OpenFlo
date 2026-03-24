# -*- coding: utf-8 -*-
"""
DOM State Hasher

Generates a stable structural fingerprint of the visible, interactive DOM.

Structural attributes (tag, id, classes, aria-label, role, type, href path)
are collected for all interactive elements. Additionally, the visible text
content of buttons, labels, and form legends is included — these are written
by UI developers and are stable, making them useful for differentiating states
with identical structure (e.g. a login form vs. a signup form, or an
"Add to Cart" button vs. a "Checkout" button).

All other text content is excluded to avoid hash churn from dynamic values
like timestamps, prices, product names, and user-specific data.
"""

import hashlib
import logging

logger = logging.getLogger(__name__)

# JavaScript snippet to extract the interactive DOM skeleton.
# Includes stable semantic text (button labels, form labels) alongside
# structural attributes, but excludes all other text content.
_DOM_SKELETON_JS = """
() => {
    const INTERACTIVE_SELECTOR = [
        'button', 'input', 'select', 'textarea',
        'a[href]', '[onclick]',
        '[role="button"]', '[role="link"]', '[role="tab"]',
        '[role="menuitem"]', '[role="checkbox"]', '[role="radio"]',
        '[role="combobox"]', '[role="textbox"]', '[role="searchbox"]'
    ].join(', ');

    // Tags whose visible text is stable developer-authored copy (safe to include)
    const STABLE_TEXT_TAGS = new Set(['BUTTON', 'LABEL', 'LEGEND']);

    // Dynamic attribute patterns to strip from class names
    const DYNAMIC_ATTR_PATTERNS = [
        /timestamp/i, /session/i, /token/i, /nonce/i,
        /random/i, /uid/i, /uuid/i, /csrf/i
    ];

    const elements = Array.from(document.querySelectorAll(INTERACTIVE_SELECTOR));
    const tuples = [];

    for (const el of elements) {
        const tag = (el.tagName || '').toLowerCase();
        const id = (el.id || '').trim();
        const classes = Array.from(el.classList || [])
            .filter(c => !DYNAMIC_ATTR_PATTERNS.some(p => p.test(c)))
            .sort()
            .join(' ');
        const ariaLabel = (el.getAttribute('aria-label') || '').trim();
        const role = (el.getAttribute('role') || '').trim();
        const type = (el.getAttribute('type') || '').trim();

        // Only include pathname (not full href with query params or hashes)
        let hrefPath = '';
        try {
            const href = el.getAttribute('href') || '';
            if (href && !href.startsWith('#') && !href.startsWith('javascript')) {
                const url = new URL(href, window.location.href);
                hrefPath = url.pathname;
            }
        } catch (e) {}

        // Include visible text only for stable developer-authored elements
        // (buttons, form labels, fieldset legends). All other text excluded
        // to prevent hash churn from dynamic content.
        let stableText = '';
        if (STABLE_TEXT_TAGS.has(el.tagName)) {
            stableText = (el.textContent || '').trim().replace(/\\s+/g, ' ');
        }

        tuples.push([tag, id, classes, ariaLabel, role, type, hrefPath, stableText].join('|'));
    }

    tuples.sort();
    return window.location.pathname + '\\n' + tuples.join('\\n');
}
"""

_SENTINEL_NO_PAGE = "__no_page__"
_SENTINEL_ERROR = "__hash_error__"


async def hash_dom_state(page) -> str:
    """
    Hash the structural skeleton of the interactive DOM.

    Extracts interactive elements (buttons, inputs, links, etc.) and hashes
    their structural attributes plus stable semantic text (button labels, form
    labels, fieldset legends) into a 16-char hex string.

    Stable text is included to differentiate pages with identical DOM structure
    but different purpose (e.g. login vs. signup form, "Add to Cart" vs.
    "Checkout" button). All other text is excluded to avoid hash churn from
    dynamic values like timestamps, prices, and user-specific content.

    Args:
        page: Playwright Page object (or None)

    Returns:
        16-char hex hash string, or a sentinel string on error/no-page.
    """
    if page is None:
        return _SENTINEL_NO_PAGE

    try:
        if page.is_closed():
            return _SENTINEL_NO_PAGE
    except Exception:
        return _SENTINEL_NO_PAGE

    try:
        dom_string = await page.evaluate(_DOM_SKELETON_JS)
        if not isinstance(dom_string, str):
            dom_string = str(dom_string)
        return hashlib.sha256(dom_string.encode("utf-8")).hexdigest()[:16]
    except Exception as e:
        logger.debug(f"DOM hashing failed: {e}")
        return _SENTINEL_ERROR


def is_sentinel(hash_value: str) -> bool:
    """Return True if the hash is a sentinel (error/no-page) value."""
    return hash_value in (_SENTINEL_NO_PAGE, _SENTINEL_ERROR)
