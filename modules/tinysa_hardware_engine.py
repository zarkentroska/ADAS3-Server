import json
import struct
import time

import numpy as np
import requests


def find_tinysa_port(list_ports_module, vid=0x0483, pid=0x5740):
    """Busca el puerto serie del TinySA por VID/PID."""
    ports = list_ports_module.comports()
    for port in ports:
        if port.vid == vid and port.pid == pid:
            return port.device
    return None


def send_tinysa_command(base_url, command_json, timeout=10):
    """Envía un comando JSON al servidor Android para controlar TinySA."""
    try:
        command_url = base_url + "/tinysa/command"
        response = requests.post(
            command_url,
            json=command_json,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if response.status_code == 200:
            print(f"[TINYSA] Comando enviado: {command_json.get('action', 'unknown')}")
            return True
        print(f"[TINYSA] Error enviando comando: HTTP {response.status_code}")
        return False
    except requests.exceptions.Timeout:
        print("[TINYSA] Timeout enviando comando (puede que el servidor no esté respondiendo)")
        return False
    except Exception as e:
        print(f"[TINYSA] Error enviando comando: {e}")
        return False


def run_tinysa_hardware_worker_serial(
    get_running_fn,
    get_sequence_fn,
    get_sequence_index_fn,
    set_sequence_index_fn,
    set_current_label_fn,
    get_serial_fn,
    set_data_ready_fn,
    tinysa_points_default,
    sweeps_per_range_default,
):
    """Worker de hardware TinySA en modo serial directo."""
    print("[TINYSA] Hardware Worker Serial iniciado")

    sequence = get_sequence_fn()
    if not sequence:
        print("[TINYSA] Sin secuencia activa, saliendo worker.")
        return

    try:
        serial_conn = get_serial_fn()
        if serial_conn is None or not serial_conn.is_open:
            print("[TINYSA] Puerto serie no disponible en hardware worker.")
            return

        serial_conn.reset_input_buffer()
        serial_conn.write(b"abort\r")
        try:
            serial_conn.read_until(b"ch> ")
        except Exception:
            pass
        time.sleep(0.05)

        while get_running_fn() and get_sequence_fn():
            sequence = get_sequence_fn()
            seq_idx = get_sequence_index_fn()
            config = sequence[seq_idx]
            start = int(config["start"])
            stop = int(config["stop"])
            points = int(config.get("points", tinysa_points_default))
            sweeps_target = max(1, int(config.get("sweeps", sweeps_per_range_default)))
            set_current_label_fn(config.get("label", ""))

            cmd = f"scanraw {start} {stop} {points}\r".encode()
            sweeps_done = 0

            while get_running_fn() and sweeps_done < sweeps_target:
                serial_conn.write(cmd)
                try:
                    raw_block = serial_conn.read_until(b"}")
                except Exception as e:
                    print(f"[TINYSA] Error leyendo bloque scanraw: {e}")
                    time.sleep(0.05)
                    continue

                if not raw_block:
                    time.sleep(0.02)
                    continue

                start_idx = raw_block.find(b"{")
                end_idx = raw_block.rfind(b"}")
                if start_idx == -1 or end_idx <= start_idx + 1:
                    time.sleep(0.02)
                    continue

                data_bytes = raw_block[start_idx + 1 : end_idx]
                if len(data_bytes) < 30:
                    time.sleep(0.02)
                    continue

                n_points = len(data_bytes) // 3
                if len(data_bytes) % 3 != 0:
                    data_bytes = data_bytes[: n_points * 3]

                if n_points != points:
                    print(
                        f"[TINYSA] Aviso: dispositivo devolvió {n_points} puntos "
                        f"en lugar de {points}."
                    )

                try:
                    values = [v[0] for v in struct.iter_unpack("<xH", data_bytes)]
                    if len(values) != n_points:
                        time.sleep(0.02)
                        continue
                    levels = (np.asarray(values, dtype=np.float32) / 32.0) - 174.0
                    freqs_dynamic = np.linspace(start, stop, n_points, dtype=np.float32)
                    set_data_ready_fn(freqs_dynamic, levels)
                except Exception as e:
                    print(f"[TINYSA] Error parseando datos scanraw: {e}")
                    time.sleep(0.02)
                    continue

                try:
                    serial_conn.read_until(b"ch> ")
                except Exception:
                    pass

                sweeps_done += 1

            set_sequence_index_fn((seq_idx + 1) % len(sequence))

    except Exception as e:
        print(f"[TINYSA] Error crítico en hardware worker: {e}")
    finally:
        set_current_label_fn("")

    print("[TINYSA] Hardware Worker Serial finalizado")


def run_tinysa_hardware_worker_http(
    get_running_fn,
    set_running_fn,
    get_use_http_fn,
    set_use_http_fn,
    get_last_sequence_payload_fn,
    send_command_fn,
    base_url,
    headers,
    connect_timeout,
    read_timeout,
    stream_chunk_size,
    get_http_response_fn,
    set_http_response_fn,
    set_data_ready_fn,
    set_current_label_fn,
):
    """Worker de hardware TinySA usando stream HTTP desde Android."""
    print("[TINYSA] Hardware Worker HTTP iniciado")
    print(f"[TINYSA] HTTP timeouts -> connect: {connect_timeout}s, read: {read_timeout}s")

    def restart_remote_scanning(reason):
        if not get_running_fn() or not get_use_http_fn():
            return None
        payload = get_last_sequence_payload_fn()
        if not payload:
            return None
        print(f"[TINYSA] Reiniciando barrido remoto ({reason})")
        send_command_fn({"action": "stop"})
        send_command_fn({"action": "set_sequence", "sequence": payload})
        send_command_fn({"action": "start"})
        return time.time()

    try:
        data_url = base_url + "/tinysa/data"
        print(f"[TINYSA] Conectando a {data_url}...")

        response = requests.get(
            data_url,
            stream=True,
            headers=headers,
            timeout=(connect_timeout, read_timeout),
        )
        set_http_response_fn(response)

        if response.status_code != 200:
            print(f"[TINYSA] Error conectando: HTTP {response.status_code}")
            set_running_fn(False)
            set_use_http_fn(False)
            return

        print("[TINYSA] Conectado al stream de datos")
        response.raw.decode_content = True
        buffer = ""
        last_data_time = time.time()

        while get_running_fn():
            try:
                chunk = response.raw.read(stream_chunk_size)
                if not chunk:
                    if get_running_fn():
                        print("[TINYSA] Stream cerrado, reintentando...")
                        time.sleep(1)
                        try:
                            response.close()
                        except Exception:
                            pass
                        response = requests.get(
                            data_url,
                            stream=True,
                            headers=headers,
                            timeout=(connect_timeout, read_timeout),
                        )
                        set_http_response_fn(response)
                        if response.status_code != 200:
                            break
                        buffer = ""
                        ts = restart_remote_scanning("reconexión tras stream cerrado")
                        last_data_time = ts if ts else time.time()
                        continue
                    break

                buffer += chunk.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if not line.startswith("{") or not line.endswith("}"):
                            continue
                        data = json.loads(line)
                        freqs_array = data.get("freqs", [])
                        levels_array = data.get("levels", [])
                        if len(freqs_array) > 0 and len(levels_array) > 0:
                            freqs = np.array(freqs_array, dtype=np.float32)
                            levels = np.array(levels_array, dtype=np.float32)
                            print(f"[HTTP {time.time():.2f}] Datos: {len(freqs)} pts")
                            set_data_ready_fn(freqs, levels)
                            last_data_time = time.time()
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"[TINYSA] Error procesando datos: {e}")
                        continue

                if time.time() - last_data_time > max(12.0, read_timeout * 2):
                    ts = restart_remote_scanning("timeout sin datos")
                    last_data_time = ts if ts else time.time()

            except requests.exceptions.RequestException as e:
                if get_running_fn():
                    print(f"[TINYSA] Error en stream: {e}, reintentando...")
                    time.sleep(1)
                    try:
                        response.close()
                    except Exception:
                        pass
                    try:
                        response = requests.get(
                            data_url,
                            stream=True,
                            headers=headers,
                            timeout=(connect_timeout, read_timeout),
                        )
                        set_http_response_fn(response)
                        if response.status_code != 200:
                            break
                        buffer = ""
                        ts = restart_remote_scanning("reconexión tras error de red")
                        last_data_time = ts if ts else time.time()
                    except Exception:
                        break
                else:
                    break
            except Exception as e:
                print(f"[TINYSA] Error inesperado: {e}")
                time.sleep(0.1)

    except Exception as e:
        print(f"[TINYSA] Error crítico en hardware worker HTTP: {e}")
        set_running_fn(False)
        set_use_http_fn(False)
    finally:
        set_current_label_fn("")
        try:
            response = get_http_response_fn()
            if response:
                response.close()
        except Exception:
            pass
        set_http_response_fn(None)

    print("[TINYSA] Hardware Worker HTTP finalizado")
