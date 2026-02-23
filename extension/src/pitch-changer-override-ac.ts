// // We override global AudioContext constructor to insert our worklet before destination

// const RealAudioContext = globalThis.AudioContext;

// function patchGlobalAudioContext() {
//     function overrideAudioContext(...args) {
//         console.log('Overriding AudioContext called with:', args);
//         const realContext = new RealAudioContext(...args);
//         return realContext;
//     }

//     overrideAudioContext.prototype = Object.create(RealAudioContext.prototype);
//     overrideAudioContext.prototype.constructor = overrideAudioContext;

//     // Override the global constructor
//     globalThis.AudioContext = overrideAudioContext;
// }

// patchGlobalAudioContext();
