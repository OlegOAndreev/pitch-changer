// Convert number of seconds into a string representation.
export function secondsToString(sec: number): string {
    const rounded = Math.max(0, Math.round(sec));
    const minutes = Math.floor(rounded / 60);
    const seconds = rounded % 60;
    if (minutes > 0) {
        return `${minutes}m${seconds}s`;
    } else {
        return `${seconds}s`;
    }
}

// Concatenate multiple Float32Arrays in one
export function concatArrays(arrays: Float32Array[]): Float32Array {
    const totalLength = arrays.reduce((sum, chunk) => sum + chunk.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (const arr of arrays) {
        result.set(arr, offset);
        offset += arr.length;
    }
    return result;
}
