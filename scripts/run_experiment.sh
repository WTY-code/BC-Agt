#!/bin/bash
set -e

# ------------------------- Configuration -------------------------
NETWORK_DIR="/root/ruc/fabric-samples/test-network"
CALIPER_DIR="/root/ruc/caliper-benchmarks"
MONITOR_SCRIPT="/root/ruc/monitor.sh"
LOG_DIR="${CALIPER_DIR}/logs/$(date +'%Y-%m-%d-%H-%M')"
MONITOR_PIDS="${LOG_DIR}/monitor_pids.txt"

# ------------------------- Monitoring Functions -------------------------
CONTAINER_LIST=()

start_monitoring() {
    echo ">>> Starting container monitoring"
    containers=$(docker ps --format '{{.Names}}' | grep -v "dev-peer")
    
    for container in $containers; do
        echo "Monitoring container: $container"
        
        # Copy monitor script to container
        if ! docker cp "${MONITOR_SCRIPT}" "${container}:/tmp/monitor.sh"; then
            echo "Error: Failed to copy monitor script to ${container}"
            exit 1
        fi

        # Start monitoring process
        docker exec -d "${container}" bash -c \
            "chmod +x /tmp/monitor.sh && /tmp/monitor.sh > /tmp/monitor.log 2>&1 & echo \$! > /tmp/monitor.pid"
        
        # Record container and host PID
        host_pid=$!
        echo "${container} ${host_pid}" >> "${MONITOR_PIDS}"
        CONTAINER_LIST+=("${container}")
    done
}

stop_monitoring() {
    echo ">>> Stopping all monitoring processes"
    while read -r container pid; do
        echo "Stopping monitoring for ${container} (PID ${pid})"
        kill -9 "${pid}" 2>/dev/null || true
        docker exec "${container}" pkill -f "monitor.sh" 2>/dev/null || true
    done < "${MONITOR_PIDS}"

    pkill -f "docker exec.*monitor.sh" 2>/dev/null || true
    rm -f "${MONITOR_PIDS}"
}

# ------------------------- Main Execution Flow -------------------------
mkdir -p "${LOG_DIR}"

echo ">>> Starting Fabric network"
cd "${NETWORK_DIR}" && ./network.sh up createChannel

echo ">>> Deploying chaincode"
cd "${NETWORK_DIR}" && ./network.sh deployCC -ccn fabcar -ccp ../../caliper-benchmarks/src/fabric/samples/fabcar/go -ccl go

start_monitoring

(
    echo ">>> Executing Caliper benchmark"
    cd "${CALIPER_DIR}" && \
    npx caliper launch manager \
        --caliper-workspace ./ \
        --caliper-networkconfig networks/fabric/test-network.yaml \
        --caliper-benchconfig benchmarks/samples/fabric/fabcar/config.yaml \
        --caliper-flow-only-test \
        --caliper-fabric-gateway-enabled 2>&1 | \
        tee >(grep -A 30 "### All test results ###" > "${LOG_DIR}/caliper.log")
) &
CALIPER_PID=$!

wait ${CALIPER_PID}

stop_monitoring

echo ">>> Collecting monitoring logs"
for container in "${CONTAINER_LIST[@]}"; do
    echo "Copying logs from ${container}"
    docker cp "${container}:/tmp/monitor.log" "${LOG_DIR}/${container}_monitor.log" 2>/dev/null || true
done

echo ">>> Compressing log files"
tar -czvf "${LOG_DIR}.tar.gz" -C "${LOG_DIR}" . > /dev/null 2>&1

echo ">>> Experiment completed. Log archive: ${LOG_DIR}.tar.gz"