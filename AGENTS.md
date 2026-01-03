* TypeScript files are located in src/
* C++ files are in wasm/src/, third party C++ libraries are in wasm/3rdparty/
* Use camelCase for C++
* Put all CSS in public/main.css and most of HTML in index.html (unless the separate page is absolutely required)
* Do not embed HTML or CSS in typescript files
* Do absolutely minimal changes to other CSS rules if you do any CSS modification
* Install JavaScript/TypeScript libraries with `npm install`
* Install C++ libraries by adding a Git submodule, unless you are using system libraries
* When changing C++ files, run `npm run build`
* When changing TypeScript or html/css files, run `npm run build:ts`
* Do not attempt opening browser for UI testing, always request the user to test the UI
