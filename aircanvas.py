import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
STYLES = ['line', 'dot', 'dash', 'square']
HELP = 'SPACE : pause   C : clear   P : color   O : style   U : undo   +/- : size   M : mask   ESC : quit'
HINT = 'L click : pen color    R click : object'
S_MIN, SMOOTH = 170, 0.7

lower = np.array([100, S_MIN, 70], np.uint8)
upper = np.array([130, 255, 255], np.uint8)
hsv = raw = None
color = tuple(int(v) for v in np.random.randint(60, 256, 3))

def on_mouse(event, x, y, flags, param):
    global lower, upper, color
    if hsv is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        color = tuple(int(v) for v in np.median(raw[y-3:y+3, x-3:x+3].reshape(-1, 3), axis=0))
    elif event == cv2.EVENT_RBUTTONDOWN:
        hh, _, vv = np.median(hsv[y-5:y+5, x-5:x+5].reshape(-1, 3), axis=0)
        lower = np.array([max(hh - 8, 0), S_MIN, max(vv - 60, 60)], np.uint8)
        upper = np.array([min(hh + 8, 179), 255, 255], np.uint8)

def scaled(text, max_w, lo=0.35, hi=0.8):
    (tw, _), _ = cv2.getTextSize(text, FONT, 1.0, 1)
    return max(lo, min(hi, max_w / tw))

def label(img, text, org, scale, shade=255):
    cv2.putText(img, text, org, FONT, scale, (shade,) * 3, 1, cv2.LINE_AA)

def draw_stroke(img, s):
    p, c, t, st = s['points'], s['color'], s['thick'], s['style']
    step = {'line': 1, 'dash': 1, 'dot': 2, 'square': 3}[st]
    for i in range(step, len(p), step):
        if st == 'dot':
            cv2.circle(img, p[i], t, c, -1)
        elif st == 'square':
            cv2.rectangle(img, (p[i][0]-t, p[i][1]-t), (p[i][0]+t, p[i][1]+t), c, -1)
        elif st == 'line' or i % 10 < 5:
            cv2.line(img, p[i-1], p[i], c, t)

cap = cv2.VideoCapture(0)
cv2.namedWindow('air canvas')
cv2.setMouseCallback('air canvas', on_mouse)

kernel = np.ones((5, 5), np.uint8)
strokes, current, smooth = [], None, None
style_idx, thickness = 0, 6
drawing, show_mask = True, False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    raw = frame.copy()
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tip = None
    if contours:
        big = max(contours, key=cv2.contourArea)
        if 300 < cv2.contourArea(big) < 20000:
            (cx, cy), r = cv2.minEnclosingCircle(big)
            tip = (int(cx), int(cy))
            cv2.circle(frame, tip, int(r) + 3, (0, 255, 0) if drawing else (0, 0, 255), 2)

    if tip and drawing:
        smooth = tip if smooth is None else tuple(smooth[i] * SMOOTH + tip[i] * (1 - SMOOTH) for i in (0, 1))
        if current is None:
            current = {'color': color, 'thick': thickness, 'style': STYLES[style_idx], 'points': []}
            strokes.append(current)
        current['points'].append((int(smooth[0]), int(smooth[1])))
    else:
        current = smooth = None

    for s in strokes:
        draw_stroke(frame, s)

    hs = scaled(HELP, w - 24)
    (_, hh_), _ = cv2.getTextSize(HELP, FONT, hs, 1)
    bar = frame.copy()
    cv2.rectangle(bar, (0, h - hh_ - 22), (w, h), (0, 0, 0), -1)
    cv2.rectangle(bar, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    cv2.rectangle(frame, (12, 9), (40, 33), color, -1)
    state = 'DRAW' if drawing else 'PAUSE'
    label(frame, f'style : {STYLES[style_idx]}    size : {thickness}    {state}', (50, 28), 0.6)

    ns = scaled(HINT, w * 0.34, 0.32, 0.5)
    (nw, _), _ = cv2.getTextSize(HINT, FONT, ns, 1)
    label(frame, HINT, (w - nw - 12, 28), ns, 185)
    label(frame, HELP, (12, h - 11), hs)

    cv2.imshow('air canvas', frame)
    if show_mask:
        cv2.imshow('mask', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        drawing = not drawing
    elif key == ord('c'):
        strokes.clear()
    elif key == ord('p'):
        color = tuple(int(v) for v in np.random.randint(60, 256, 3))
    elif key == ord('o'):
        style_idx = (style_idx + 1) % len(STYLES)
    elif key == ord('u') and strokes:
        strokes.pop()
    elif key in (ord('+'), ord('=')):
        thickness = min(thickness + 2, 30)
    elif key == ord('-'):
        thickness = max(thickness - 2, 2)
    elif key == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow('mask')
    if key in (ord('c'), ord('p'), ord('o'), ord('u')):
        current = None

cap.release()
cv2.destroyAllWindows()