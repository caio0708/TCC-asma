# -*- coding: utf-8 -*-
"""
Leitor Serial Simplificado para MAX30102
------------------------------------------
Este script foi modificado para apenas exibir os valores brutos
de IR (infravermelho) e RED (vermelho) recebidos via porta serial.
Toda a lógica de cálculo de SpO2 e BPM foi removida para facilitar
a visualização e depuração dos dados do sensor.
"""
import time
import json
import serial
import serial.tools.list_ports

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
# Altere aqui para a porta serial correta do seu dispositivo
# Você pode usar a função find_serial_port() para ajudar a encontrar
SERIAL_PORT = 'COM7'
BAUD_RATE = 115200

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================
def main():
    """
    Função principal que abre a porta serial, lê os dados,
    e imprime os valores de IR e RED.
    """
    print("="*50)
    print("Leitor de Dados Brutos (IR e RED) do Sensor")
    print("="*50)
    
    try:
        # Abre a conexão com a porta serial
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"Porta serial '{SERIAL_PORT}' aberta com sucesso a {BAUD_RATE} bps.")
            print("Aguardando dados... (Pressione Ctrl+C para sair)")
            
            # Loop infinito para ler continuamente os dados
            while True:
                try:
                    # Lê uma linha da porta serial
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Se a linha não estiver vazia, tenta processá-la
                    if line:
                        # Verifica se a linha parece ser um objeto JSON
                        if line.startswith('{') and line.endswith('}'):
                            try:
                                # Tenta decodificar a linha como JSON
                                data = json.loads(line)
                                
                                # Extrai os valores de IR e RED se existirem
                                ir_value = data.get("ir")
                                red_value = data.get("red")
                                
                                # Se ambos os valores foram encontrados, imprime na tela
                                if ir_value is not None and red_value is not None:
                                    print(f"IR: {ir_value:<8} | RED: {red_value:<8}")

                            except json.JSONDecodeError:
                                # Se a linha não for um JSON válido, informa (opcional)
                                # print(f"Linha recebida não é um JSON válido: {line}")
                                pass
                        # Você pode adicionar aqui a lógica para ler temperatura/umidade se necessário
                        # Ex: elif "Temperatura:" in line:
                        #         print(line)

                except Exception as e:
                    print(f"Ocorreu um erro durante a leitura: {e}")
                    time.sleep(1)

    except serial.SerialException as e:
        print(f"\nERRO: Não foi possível abrir a porta serial '{SERIAL_PORT}'.")
        print(f"Detalhe: {e}")
        print("Verifique se a porta está correta e se o dispositivo está conectado.")
    except KeyboardInterrupt:
        print("\n\nPrograma encerrado pelo usuário.")

# =============================================================================
# PONTO DE ENTRADA DO SCRIPT
# =============================================================================
if __name__ == "__main__":
    main()