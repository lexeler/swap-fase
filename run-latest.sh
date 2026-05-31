#!/usr/bin/env bash
# Запуск swap-fase с САМЫМ СВЕЖИМ фото из ~/Downloads,
# на цветной камере + виртуальная камера (DeepLiveCam) для созвонов.
# Использование:  ./run-latest.sh     (попросит пароль sudo для виртуалки)
set -uo pipefail
cd "$(dirname "$0")"

SITE="$(.venv/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=src

# 1) остановить старый запуск + выбрать свежее фото (+поля, если лицо обрезано)
.venv/bin/python - <<'PY'
import os, signal, glob, cv2
me, parent = os.getpid(), os.getppid()
for d in glob.glob('/proc/[0-9]*/cmdline'):
    try: pid = int(d.split('/')[2])
    except: continue
    if pid in (me, parent): continue
    try: cmd = open(d, 'rb').read().replace(b'\x00', b' ').decode('utf8', 'ignore')
    except: continue
    if 'python' in cmd and 'run.py' in cmd and '--target' in cmd:
        try: os.kill(pid, signal.SIGKILL)
        except: pass
dl = os.path.expanduser('~/Downloads'); exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
imgs = [os.path.join(dl, f) for f in os.listdir(dl) if f.lower().endswith(exts)]
imgs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
if not imgs:
    print("НЕТ ФОТО в ~/Downloads"); open('/tmp/sf_t.txt', 'w').write(''); raise SystemExit
newest = imgs[0]; print("Фото:", os.path.basename(newest))
from insightface.app import FaceAnalysis
from swapfase.bootstrap import MODELS_DIR
import swapfase.providers as pr
fa = FaceAnalysis(name="buffalo_l", root=MODELS_DIR, providers=pr.select_providers(True))
fa.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread(newest); final = newest
if img is not None and len(fa.get(img)) == 0:
    for fr in (0.4, 0.6, 0.8):
        h, w = img.shape[:2]
        pad = cv2.copyMakeBorder(img, int(h*fr), int(h*fr), int(w*fr), int(w*fr), cv2.BORDER_REPLICATE)
        if len(fa.get(pad)) > 0:
            final = '/tmp/sf_target_padded.jpg'; cv2.imwrite(final, pad)
            print("(добавил поля — лицо было обрезано)"); break
open('/tmp/sf_t.txt', 'w').write(final or '')
PY

TGT="$(cat /tmp/sf_t.txt 2>/dev/null)"
[ -z "$TGT" ] && { echo "Нет подходящего фото — положи фото лица в ~/Downloads"; exit 1; }

# 2) пересоздать виртуальную камеру (нужен sudo), чтобы writer мог в неё писать
for pid in $(fuser /dev/video10 2>/dev/null); do
  ps -o args= -p "$pid" 2>/dev/null | grep -qi chrome || kill -9 "$pid" 2>/dev/null
done
sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback 2>/dev/null || true
sleep 1

# 3) запуск: цветная камера (index 0) + виртуалка
rm -f /tmp/sf_local.log
nohup ./run.sh --target "$TGT" --vcam --device 0 >/tmp/sf_local.log 2>&1 &
sleep 11
if pgrep -af "run.py.*--vcam" | grep -q -v pgrep; then
  echo ""
  echo "✅ Запущено. Окно-превью открыто."
  echo "   • Окно НЕ закрывай — только сворачивай (закроешь → виртуалка погаснет)."
  echo "   • Для Meet: перезапусти Chrome целиком и выбери камеру 'DeepLiveCam'."
else
  echo ""
  echo "⚠ Не запустилось. Частые причины:"
  echo "   • цветная камера video0 занята Chrome — закрой вкладку с камерой и запусти снова;"
  echo "   • смотри лог: tail -n 20 /tmp/sf_local.log"
fi
