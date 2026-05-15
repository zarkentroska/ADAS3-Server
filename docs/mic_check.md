# Comprobar el array acústico (y el puente EP32 BT) desde el server

Este documento describe cómo verificar, paso a paso, que los 4 micros I2S
del array ESP32 están bien conectados y emitiendo datos hasta el server,
y cómo el servidor activa el puente Bluetooth del móvil sin escanear BT
localmente.

## 0. Endpoints expuestos por el cliente Android

Resumen del contrato HTTP que el server consume. Sustituye `PHONE` por
`<ip-del-movil>:<puerto>` (por defecto `:8080`):

| Endpoint | Método | Para qué |
|---|---|---|
| `/adas3/ep32-control` | POST | Activar/desactivar/reconectar el puente Bluetooth del móvil desde el server. Body: `{"action":"enable\|disable\|reconnect\|stop"}`. Respuesta `202` + snapshot. |
| `/adas3/ep32-status` | GET | Snapshot del puente: `connected`, `state` (OFF/SCANNING/CONNECTING/CONNECTED/ERROR), `detail`, `enabled`, `active`, `bt_adapter_enabled`, `permissions_granted`, `firmware`, `mic_count`. |
| `/adas3/ep32-command` | POST | Enviar al ESP32 una pulsación: `{"command":"UP\|DOWN\|LEFT\|RIGHT\|TEST\|STATUS"}`. |
| `/adas3/mic-array/status` | GET | Snapshot del array (último heartbeat + última `acoustic` + wiring). |
| `/adas3/mic-array/data` | GET | Stream NDJSON keep-alive de heartbeats + acoustics. |

El server (testcam.py) usa estos endpoints por debajo cuando el usuario
pulsa el botón `EP32 BT` y cuando dispara una flecha. Nunca escanea
Bluetooth local. Si el APK del móvil es antiguo y no tiene
`/adas3/ep32-control` ni `/adas3/ep32-status`, el server degrada
automáticamente al probe + `/adas3/ep32-command` (modo `legacy_bridge`).

Hay tres niveles de comprobación, del más directo al más completo:

1. Heartbeat — "¿el ESP32 ve sus dos buses I2S?"
2. Stream NDJSON — "¿llegan eventos `acoustic` cuando hago ruido?"
3. Badge en pantalla — "¿el server lo está pintando correctamente?"

Se asume:

- El cliente Android está corriendo y emparejado con `ESP32-ADAS3` por
  Bluetooth.
- En la Android: el switch `EP32 BT` está en ON y el indicador muestra
  `CONNECTED`.
- El server conoce la IP/puerto del móvil (`base_url`, p.ej.
  `http://192.168.1.42:8080`).

Sustituye `PHONE` por esa IP:puerto en los ejemplos. Necesitas `curl` y
opcionalmente `jq` para leer el JSON con más comodidad.

```bash
export PHONE=192.168.1.42:8080
```

## 0.1 Flujo de "EP32 BT ON" desde el server

Cuando el usuario pulsa el botón **EP32 BT: OFF** en la UI del server:

```
testcam.py: _handle_ep32_toggle()
   │
   ├─ ep32_controller.toggle_enabled()       (flag local → ON)
   ├─ threading.Thread:
   │     ep32_controller.request_control("enable")
   │        ├─ 202 + snapshot   → estado idle/connected/error según snapshot
   │        ├─ 404/405          → control_supported=False, fallback probe
   │        ├─ ConnectionError  → bridge_unreachable
   │     ep32_controller.fetch_status()      (refresca snapshot real)
   │
   └─ render loop sigue → _tick_ep32_status_poll() refresca cada ~1.5s
```

Y cuando se pulsa **EP32 BT: ON** (apagado):

```
   ep32_controller.toggle_enabled()                (flag local → OFF)
   ep32_controller.request_control("disable")      (puente Android también OFF)
```

Si el cliente Android es antiguo (sin `/adas3/ep32-control`), la primera
respuesta `404` deja el flag `control_supported=False` y a partir de ese
punto el server ya no insiste con `/adas3/ep32-control`: sólo manda
flechas por `/adas3/ep32-command`. Funciona, pero pierdes la posibilidad
de activar/desactivar el puente Bluetooth del móvil desde el server.

## 0.2 Comprobar manualmente el puente desde shell

```bash
export PHONE=192.168.1.42:8080

# Activar el puente Bluetooth del móvil desde el server
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"type":"adas3-ep32-control","action":"enable"}' \
  http://$PHONE/adas3/ep32-control | jq .

# Esperar a CONNECTED (polling)
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -s http://$PHONE/adas3/ep32-status | jq -r '"\(.state) connected=\(.connected) mic=\(.mic_count)"'
  sleep 1
done
```

Si el snapshot final dice `CONNECTED connected=true mic=4`, ya puedes
mandar flechas:

```bash
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"type":"adas3-ep32-command","command":"UP"}' \
  http://$PHONE/adas3/ep32-command
```

Si en su lugar ves `ERROR` con `detail: "Bluetooth is disabled"`, el
adaptador BT del móvil está apagado; si ves `permissions_granted=false`,
abre la app Android y concede los permisos BT.

## 1. Heartbeat: ¿el ESP32 reconoce sus 4 micros?

Cada segundo el firmware del ESP32 publica un `heartbeat`. El cliente
Android lo guarda y lo expone por HTTP. Pídeselo:

```bash
curl -s http://$PHONE/adas3/mic-array/status | jq .
```

Salida esperada (resumen):

```json
{
  "connected": true,
  "heartbeat": {
    "mic_count": 4,
    "firmware": "esp32-adas3 0.2.0"
  },
  "last_acoustic": null,
  "wiring": {
    "mic_count": 4,
    "buses": [
      {"pair": "A", "bclk_gpio": 14, "lrcl_gpio": 13, "dout_gpio": 34, ...},
      {"pair": "B", "bclk_gpio": 22, "lrcl_gpio": 21, "dout_gpio": 35, ...}
    ],
    ...
  }
}
```

Lo importante:

- `connected: true` → el móvil tiene SPP abierto con el ESP32.
- `heartbeat.mic_count: 4` → los **dos buses** I2S están vivos en el
  ESP32. Si pone `2`, el firmware sólo ha podido inicializar una de las
  parejas (revisa `dout_gpio` 34/35 y SEL de cada mic). Si pone `0`,
  ningún I2S está vivo.
- `wiring.buses[].dout_gpio` debe ser `34` (pareja A) y `35` (pareja B).
  Si no coinciden, el firmware flasheado **no es el unificado** o se
  flasheó otra variante.

Diagnóstico rápido de cableado por valor de `mic_count`:

| `mic_count` | Probable causa |
|---|---|
| `0` | ESP32 no está alimentando 3V3 a los mics, o ambos DOUT desconectados |
| `2` | Una pareja sin alimentación, SEL flotante, o DOUT mal soldado |
| `4` | Todo OK desde el punto de vista I2S |

## 2. Stream: ¿llegan eventos cuando haces ruido?

El cliente Android expone un canal NDJSON keep-alive. Conéctate y haz
ruido cerca del array (palmada fuerte, voz, silbido):

```bash
curl -N -s http://$PHONE/adas3/mic-array/data | head -20
```

Esperas:

- Una línea `heartbeat` cada ~1 s (siempre).
- Líneas `acoustic` cuando superas el umbral. Ejemplo:

```json
{"type":"heartbeat","mic_count":4,"firmware":"esp32-adas3 0.2.0",...}
{"type":"acoustic","detected":true,"doa_deg":24.3,"energy":0.082,"confidence":0.78,"mic_count":4,"pair":""}
```

Si **sólo** ves heartbeats y nunca `acoustic`, es uno de:

- Los mics no captan nada (cableado SEL, masa, 3V3).
- El umbral del firmware está demasiado alto. Edita
  `ENERGY_THRESHOLD` (0.05 por defecto) y `CONFIDENCE_THRESHOLD`
  (0.55) en `firmware/esp32-adas3/esp32-adas3.ino` y reflashea.
- Estás susurrando. Prueba con una palmada seca a ~30 cm.

Si ves líneas `acoustic` pero `doa_deg` siempre 0 o ±90 fijo, recuerda
que la DOA del firmware es **placeholder L/R balance**, no GCC-PHAT
real; eso no significa que los mics estén rotos.

## 3. Forzar un heartbeat al ESP32 desde el server

Útil para confirmar el camino completo
`server → Android → SPP → ESP32 → SPP → Android → server`:

```bash
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"type":"adas3-ep32-command","command":"STATUS"}' \
  http://$PHONE/adas3/ep32-command
```

Esperado: `200 {"status":"received"}`. En `/adas3/mic-array/data` debes
ver un `heartbeat` fuera de cadencia justo a continuación.

Si en su lugar recibes `409 {"status":"not_connected"}`, el puente
HTTP funciona pero el ESP32 no está realmente emparejado en el móvil —
abre la app Android, comprueba que el switch `EP32 BT` está en ON y
que el indicador muestra `CONNECTED`.

## 4. Comprobación visual en el server: dos badges, dos cosas distintas

Con el server `testcam.py` corriendo, cuando activas `EP32 BT: ON` y la
fuente de audio es `esp32_array`, aparecen **dos badges separados**
debajo del D-pad EP32:

### 4.1 `ARRAY DIR [<transport>]` — telemetría direccional

**NO es audio audible.** Es lo que el ESP32 calcula y manda por
Bluetooth como JSONL: estimación de dirección + energía/confianza
agregadas a partir de las parejas I2S.

```
ARRAY DIR [serial]
DOA~: +1.8 deg
E:0.12  C:0.58  mic:4  pair:A
        + círculo con una "rebanada" que apunta a +1.8°
```

Significado de cada campo:

| Campo | Qué es | Rango |
|---|---|---|
| `transport` | Cómo viaja la telemetría: `serial`, `bluetooth`, `simulation`. Con el array real por Bluetooth deberías ver `bluetooth` o `serial` (modo USB-CDC). `simulation` significa que el cliente del array está en modo demo. | string |
| `DOA~` (con `~`) | **Dirección de llegada (ESTIMADA)** del sonido más fuerte respecto al frente del array. 0° = frente; positivo = derecha. La tilde `~` está adrede: es un **placeholder por balance L/R**, NO GCC-PHAT real. La precisión es coarse (±20-30°). | -180° a +180° |
| `E` | Energía normalizada estimada por el firmware (0..1 aprox.). No es dBFS de un PCM, es la métrica que el ESP32 publica en cada evento `acoustic`. | 0.0 - 1.5 |
| `C` | Confianza del detector LOCAL del ESP32 (no Keras). Sube cuando el ratio energía/umbral cruza `CONFIDENCE_THRESHOLD`. | 0.0 - 1.0 |
| `mic` | Número de micrófonos del array activos según el último heartbeat. Lo esperado es `4`. | 1-8 |
| `pair` | Pareja I2S de la que vino la última muestra (`A` = Mic1/Mic2 GPIO14/13/34; `B` = Mic3/Mic4 GPIO22/21/35). Útil para confirmar que la otra pareja también está viva — debería ir alternando si las dos parejas están bien. | A / B |

El **círculo con "rebanada"** es una flecha de DOA, no un VU meter.
El círculo es la circunferencia 360° y la línea va del centro hacia
la dirección estimada (0° = arriba, 90° = derecha, -90° = izquierda).
**No** indica nivel de audio.

Colores del título:
- **Verde** `ARRAY DIR [...]` → array conectado, idle (sin detección
  local). Los heartbeats llegan.
- **Ámbar** `ARRAY DIR DETECT` → el ESP32 está reportando una
  detección. Esto **no implica** que Keras haya detectado un dron —
  son detectores independientes.
- **Rojo** `ARRAY DIR OFF` → no llegan heartbeats al server.

### 4.2 `AUDIO ARRAY` — nivel de PCM crudo

**Esto sí es audio.** Se dibuja sólo si la fuente activa es
`esp32_array` y el bridge HTTP está streaming. Reemplaza por completo
al antiguo `E/C` del badge direccional como fuente de verdad sobre
"está entrando audio útil al servidor".

```
AUDIO ARRAY
1ch @ 44100 Hz                              rms:412 pk:8740
[████████████░░░░░░░░░░░░░]
```

| Campo | Qué es |
|---|---|
| `1ch @ 44100 Hz` | Sample rate / canales que el bridge realmente recibe. Coincide con la cabecera HTTP `Content-Type` de `/adas3/mic-array/pcm`. |
| `rms` | Root mean square del último chunk PCM, en escala int16 absoluta (0..32767). 0 = silencio absoluto. ~5000-10000 = nivel de habla normal. |
| `pk` | Pico absoluto del último chunk. Si `pk < 200` durante 20 chunks seguidos, el título cambia a `AUDIO ARRAY SILENT` en rojo: el array está mudo aunque el bridge esté recibiendo bytes (cableado SEL, GND, 3V3 o el firmware no hizo AUDIO_ON). |
| Barra horizontal | Visualización rápida del peak (0..32767 → 0..ancho del badge). |

**Si oyes ruido por los altavoces y rms/pk se mueven con tu voz, el
array está captando audio útil que Keras puede analizar.** Si no
oyes nada y `pk` está pegado a 0, el problema NO es el bridge — el
PCM está llegando vacío.

### 4.3 Cuando NO ves los badges

- `ARRAY DIR` ausente con EP32 BT en ON → el server no recibe
  heartbeats. Mira `_get_ep32_status_text()` (debe estar en
  `connected`, no `bridge_unreachable`).
- `AUDIO ARRAY` ausente con `audio_source=esp32_array` → el bridge
  no está en estado `streaming`. Causas: ni `/adas3/mic-array/pcm`
  ni el fallback `/audio` responden (state `not_implemented`), BT
  desconectado, o `array_audio_frames_forwarded` = 0 en el móvil.
  Si `/mic-array/pcm` devuelve 404, el server reintenta en `/audio`
  automáticamente (`using_fallback: true` en `get_state()`).

## 5. Resumen de diagnóstico exprés

### 5.1 Telemetría / heartbeat

| Síntoma | Empieza por |
|---|---|
| `mic_count: 0` en heartbeat | Cableado 3V3/GND y bus I2S, monitor serie del ESP32 |
| `mic_count: 2` en heartbeat | Pareja B (DOUT=35, BCLK=22, LRCL=21), SEL Mic3/Mic4 |
| `mic_count: 4` pero sin `acoustic` | Mics OK; umbrales en firmware demasiado altos |
| Stream vacío | `EP32 BT` en OFF en la app Android, o `bridge_unreachable` en el server |
| Server no pinta badge `ARRAY DIR` | `ep32_controller.is_enabled()` debe estar en `idle`/`connected`; mira `/get_state()` |

### 5.2 Audio PCM (la rama crítica para Keras)

| Síntoma | Qué está pasando | Empieza por |
|---|---|---|
| Selector dice `esp32_array`, kbps suben, **no se oye nada** | Antes de esta revisión el playback sólo funcionaba para `phone_mic`. **Actualiza testcam.py + acoustic_integration.py + modules/array_audio_bridge.py** | reflashea/sincroniza |
| `AUDIO ARRAY SILENT` (rojo), pk=0 | Bridge recibe bytes pero todos son cero/ruido | Pareja A SEL/3V3/GND, firmware `AUDIO_INT16_SHIFT` muy alto (sube a 12 o 10), comando `AUDIO_ON` no enviado |
| `AUDIO ARRAY`, pk sube con voz pero suena distorsionado | Clipping en el firmware: subiste demasiado la ganancia | Baja `AUDIO_INT16_SHIFT` (más shift = menos ganancia) en el firmware o `software_gain` en el bridge |
| Audio entrecortado, FPS bien | Cola Keras (`audio_buffer`) muy pequeña respecto a la cadencia del bridge | Sube `audio_buffer maxsize` en testcam o baja el chunk_size del bridge |
| FPS caen al activar array | Bridge demasiado ávido en CPU | Sube `chunk_size` (menos `iter_content` calls) o baja `AUDIO_OUT_RATE_HZ` en firmware |

### 5.3 ¿Realmente detecta drones a mayor distancia?

El array NO mejora intrínsecamente el alcance Keras respecto al
micrófono del móvil **a menos que** lo coloques cerca del cielo /
lejos del operador, con menos viento y reflexiones. Con el modelo
Keras actual entrenado para el micro del móvil, vas a observar:

| Distancia dron | Cosa que esperar |
|---|---|
| < 30 m | Detecciones consistentes con cualquiera de las dos fuentes. Sirve para validar que el array está conectado y entregando audio útil. |
| 30-80 m | El array empieza a aportar valor si lo orientas hacia el dron (efecto direccional placeholder + menos ruido del operador). La detección Keras sube en `confidence` en algunos eventos, baja en otros. |
| > 80 m | El factor limitante deja de ser el micro y empieza a ser la SNR del entorno. Sin reentrenar Keras con muestras del array no esperes mejora dramática. |

Para una comparación clara con grabaciones de referencia:

1. Coloca el móvil y el array a ~50 m del dron en línea recta.
2. Activa `audio_source=phone_mic`, anota `confidence` y `is_drone`
   durante 30 s.
3. Activa `audio_source=esp32_array` sin mover nada, anota lo mismo.
4. Compara: si el array baja la confianza media en >30% probablemente
   el firmware emite muy bajo (sube ganancia) o las parejas I2S no
   están orientadas hacia el dron.

## 6. Comprobación end-to-end automatizada (opcional)

Pequeño script bash que combina todo lo anterior:

```bash
#!/usr/bin/env bash
set -e
: "${PHONE:?Set PHONE=ip:port}"
echo "[1/3] mic-array status"
curl -s http://$PHONE/adas3/mic-array/status | jq '{
  connected,
  mic_count: .heartbeat.mic_count,
  firmware: .heartbeat.firmware,
  dout: [.wiring.buses[].dout_gpio]
}'

echo "[2/3] forzar STATUS al ESP32"
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"type":"adas3-ep32-command","command":"STATUS"}' \
  http://$PHONE/adas3/ep32-command | jq .

echo "[3/3] capturando 5 segundos de stream"
timeout 5 curl -N -s http://$PHONE/adas3/mic-array/data | head -10
```

Una sesión sana imprime `mic_count: 4`, `dout: [34, 35]`, recibe
`{"status":"received"}` y vuelca varias líneas de `heartbeat` /
`acoustic`.
