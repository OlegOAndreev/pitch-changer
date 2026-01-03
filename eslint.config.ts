import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import css from "@eslint/css";
import { defineConfig } from "eslint/config";

export default defineConfig([
  { files: ["**/*.{js,mjs,cjs,ts,mts,cts}"], plugins: { js }, extends: ["js/recommended"], languageOptions: { globals: globals.browser } },
  { files: ["**/*.{ts,mts}"], plugins: { js }, extends: ["js/recommended"], languageOptions: { globals: globals.browser } },
  tseslint.configs.recommended,
  { files: ["**/*.css"], plugins: { css }, language: "css/css", extends: ["css/recommended"] },

  {
    rules: {
      // Allow unused variables starting with exactly one underscore.
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { "argsIgnorePattern": "^_[^_].*$|^_$", "varsIgnorePattern": "^_[^_].*$|^_$", "caughtErrorsIgnorePattern": "^_[^_].*$|^_$" }
      ]
    }
  }
]);
