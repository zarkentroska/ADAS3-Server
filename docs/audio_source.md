# Selector de origen de audio para Keras

## Resumen

ADAS3 tiene dos rutas físicas para alimentar el modelo Keras de detección
acústica de drones:

| Origen | Identificador | Captura | Transporte |
|---|---|---|---|
| Micrófono del móvil | `phone_mic` (default) | mic interno Android | HTTP `GET /audio` que ya expone el cliente Android |
| Array ESP32 (4 micros I2S) | `esp32_array` | array MEMS conectado al ESP32 | HTTP `GET /adas3/mic-array/pcm` (contrato definido aquí, **a implementar en el cliente Android**) |

El usuario alterna entre ambos desde la UI con el botón izquierdo
`AUDIO: MIC MOVIL / AUDIO: ARRAY ESP32` (o programáticamente con
`toggle_audio_source()`). El motor Keras NO sabe — ni debe saber —
cuál de los dos está activo: ambos alimentan el mismo
`audio_buffer: queue.Queue` que consume
`run_audio_detection_worker`.

## Contrato compartido (lo que entra a `audio_buffer`)

Ambas fuentes deben empujar PCM en el siguiente formato:

| Campo | Valor |
|---|---|
| Endianness | little-endian |
| Dtype | `int16` |
| Canales | 1 (mono) por defecto |
| Sample rate | 44100 Hz por defecto |
| Tamaño de chunk | libre (>=512 bytes recomendado) |

El sample rate / canales pueden negociarse via cabeceras HTTP:

```
Content-Type: audio/pcm; rate=<HZ>; channels=<N>
```

El parser está en dos sitios (mismo formato):
- `testcam.stream_audio` para `/audio` (phone_mic, ya implementado).
- `modules.array_audio_bridge.ArrayAudioBridge._parse_content_type`
  para `/adas3/mic-array/pcm` (esp32_array, nuevo).

Si el ESP32 acaba enviando PCM a 16 kHz mono para ahorrar ancho de
banda Bluetooth (recomendado), basta con que el cliente Android
anuncie `Content-Type: audio/pcm; rate=16000; channels=1`. El bridge
servidor lo absorbe; ahora bien, el motor Keras se entrenó con
**44100 Hz**, así que **el cliente Android es el responsable de
resamplear a 44100 Hz antes de servir** si el ESP32 capta a otro
rate. Esa es la decisión más segura para evitar tocar el modelo.

Variante alternativa NDJSON (también soportada):

```
Content-Type: application/x-ndjson
Body (una linea por chunk):
{"b64":"<base64 de int16 LE PCM>","seq":<int>,"rate":44100,"channels":1}
```

El bridge decodifica `b64` con `base64.b64decode` y empuja los bytes
crudos al `audio_buffer`. `seq`/`rate`/`channels` son informativos.
Esta variante se mantiene como red de seguridad por si el cliente
Android decide multiplexar metadatos (timestamp, pair) en la misma
conexión que el audio.

## Contrato del lado cliente Android (a implementar)

El cliente Android **aún no expone** `/adas3/mic-array/pcm`. Cuando lo
implemente, debe:

1. Captar el audio del ESP32 por Bluetooth SPP. El firmware unificado
   (`firmware/esp32-adas3/esp32-adas3.ino`) ya muestrea I2S y publica
   eventos `acoustic`/`heartbeat`. Para alimentar a Keras tiene que
   añadir un canal extra de **PCM crudo** (o `b64` por NDJSON) — el
   firmware actual NO lo hace todavía. Es un paso futuro.
2. Resamplear de 16 kHz (rate del array MEMS) a 44100 Hz antes de
   exponer al server. El móvil tiene `MediaCodec`/`AudioTrack` para
   hacerlo barato.
3. Cumplir el `Content-Type: audio/pcm; rate=44100; channels=1`.
4. Manejar el caso "ESP32 no conectado" devolviendo `503` o
   `409`, igual que ya hace `/adas3/ep32-command`. El bridge del
   server marcará entonces `state=error` y se reconectará.
5. Si el endpoint dedicado devuelve `404`, el servidor **reintenta en
   `/audio`** (mismo PCM que Android ya expone con `esp32_array`).
   Solo si ambos fallan, el bridge pasa a `not_implemented`.

## ¿Cómo distinguir "ARRAY OK direccional" de "ARRAY OK audio Keras"?

El sistema acústico tiene **dos roles distintos**:

| Rol | Qué publica | Endpoint cliente | Estado en server |
|---|---|---|---|
| Sensor direccional | `heartbeat`/`acoustic` (DOA, energía, confianza) | `/adas3/mic-array/data` (NDJSON), `/adas3/mic-array/status` (snapshot) | `acoustic_state()`, badge `ARRAY OK` |
| Audio PCM para Keras | PCM int16 LE @ 44100 Hz | `/adas3/mic-array/pcm` | `array_audio_bridge.get_state()` |

Si quieres saber cuál está activo:

```bash
PHONE=192.168.1.42:8080

# 1. Direccional (siempre disponible si el ESP32 está conectado):
curl -s http://$PHONE/adas3/mic-array/status | jq .
# → {connected:true, heartbeat:{mic_count:4,...}, ...}

# 2. Audio PCM (puede no estar implementado aún):
curl -sI http://$PHONE/adas3/mic-array/pcm
# → 200 OK con Content-Type: audio/pcm... → disponible
# → 404 Not Found → el server usará /audio como fallback automático
```

En el server:

- El badge `ARRAY OK [transport]` debajo del D-pad EP32 indica el
  **sensor direccional** (DOA, mic_count, energía). Significa: la
  cadena `ESP32 → Android → server` está viva *para eventos*. No
  garantiza que haya un stream PCM disponible.
- El **botón "AUDIO: ARRAY ESP32"** con sufijo `[streaming]` indica el
  **audio PCM** entrando al pipeline Keras. Si pone `[off]` /
  `[connecting]` / `[error]` / `[not_implemented]`, el stream PCM no
  está alimentando al modelo.

Cuando el usuario tiene `AUDIO: ARRAY ESP32 [streaming]`, los chunks
PCM están llegando al `audio_buffer` y Keras los está analizando. El
modelo NO ve los heartbeats/acoustic; esos siguen su flujo en
`acoustic_integration.py` y disparan el evento interno
`acoustic_array` por separado.

## Smoke test rápido en el server

```python
# Desde un REPL con testcam.py corriendo:
from testcam import (
    audio_source_controller, array_audio_bridge,
    set_audio_source,
)
set_audio_source("esp32_array")
print(array_audio_bridge.get_state())
# state esperado: "starting" → "connecting" → "streaming"
#                 si /adas3/mic-array/pcm responde 200.
# o:              "not_implemented" si responde 404 (el caso actual).
```

## Por qué NO se mezclan los dos sensores

`acoustic_array` (DOA/energía) y `audio_detection` (Keras) tienen
escalas, debounces y consumidores diferentes:

- `acoustic_array` dispara el evento interno hacia el cliente Android
  para que sepa de dónde viene el ruido (no manda Telegram).
- `audio_detection` (Keras) sí manda Telegram cuando confirma drone
  durante N ventanas consecutivas.

Si ambos se gatillaran en cascada con el mismo PCM, una sola palmada
podría producir un Telegram + un acoustic_array + un drone confirmado
mal calibrado. Por eso el contrato dice **el array entra a Keras pero
NO dispara Telegram por sí mismo** — sólo a través de la cadena
Keras → confirm window → Telegram, igual que el mic del móvil.
