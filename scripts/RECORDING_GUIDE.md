# 🎬 Как записать GIF-демку для README

## Подготовка терминала

1. **Шрифт:** Увеличь размер шрифта до **16-18pt** (⌘+ несколько раз)
2. **Размер окна:** Растяни на ~100 символов в ширину, 30 строк в высоту
3. **Тема:** Используй тёмную тему (Pro или Homebrew в Terminal.app)
4. **Очисти историю:** `clear`

---

## Способ 1: VHS (Charmbracelet) — автоматическая запись ⭐ РЕКОМЕНДУЮ

Самый красивый результат. Воспроизводит скрипт автоматически.

```bash
# Установка
brew install charmbracelet/tap/vhs

# Запуск
cd /Users/stanislav/Desktop/NAP/nra
vhs scripts/demo.tape
```

Создай файл `scripts/demo.tape`:
```tape
Set Shell "bash"
Set FontSize 16
Set Width 1000
Set Height 600
Set Theme "Catppuccin Mocha"
Set Padding 20

Output docs/assets/demo.gif

Type "python scripts/record_demo.py"
Enter
Sleep 20s
```

---

## Способ 2: asciinema + agg — ручная запись

```bash
# Установка
brew install asciinema
cargo install --git https://github.com/asciinema/agg

# Запись (ты вручную запускаешь скрипт)
cd /Users/stanislav/Desktop/NAP/nra
asciinema rec demo.cast

# >>> В терминале запусти:
# python scripts/record_demo.py
# >>> Когда скрипт завершится, нажми Ctrl+D

# Конвертация в GIF
agg demo.cast docs/assets/demo.gif --theme monokai --font-size 16
```

---

## Способ 3: QuickTime + ffmpeg — screen capture

```bash
# 1. Открой QuickTime Player → File → New Screen Recording
# 2. Выбери область терминала
# 3. Запусти скрипт: python scripts/record_demo.py
# 4. Останови запись
# 5. Сохрани как demo.mov

# Конвертация в GIF (ffmpeg)
brew install ffmpeg
ffmpeg -i demo.mov -vf "fps=15,scale=800:-1" -gifflags +transdiff docs/assets/demo.gif

# Оптимизация размера (если >5MB)
brew install gifsicle
gifsicle -O3 --lossy=80 docs/assets/demo.gif -o docs/assets/demo.gif
```

---

## После записи

GIF должен оказаться в `docs/assets/demo.gif`. Потом мы добавим его в README:

```markdown
<div align="center">
  <img src="docs/assets/demo.gif" alt="NRA Demo" width="800"/>
</div>
```

---

## Чеклист перед записью

- [ ] Активируй venv: `source nra-python/.venv/bin/activate`
- [ ] Проверь что `import nra` работает
- [ ] Убедись что есть интернет (скрипт ходит на HuggingFace)
- [ ] Закрой лишние вкладки/уведомления (чтобы не попали в кадр)
- [ ] Шрифт 16-18pt, тёмная тема
