const path = require('path');
const version = require('./package.json').version;

// Custom webpack rules
const rules = [
    { test: /\.ts$/, loader: 'ts-loader' },
    { test: /\.js$/, loader: 'source-map-loader' },
    { test: /\.css$/, use: [
            // Not sure why it's necessary to not include these loaders.
            // It looks like JupyterLab is automatically including these somewhere, so it's an error to duplicate
            // them
            'style-loader', 'css-loader'
        ]
    }
];

// Packages that shouldn't be bundled but loaded at runtime
const externals = ['@jupyter-widgets/base'];

const resolve = {
    // Add '.ts' and '.tsx' as resolvable extensions.
    extensions: [".webpack.js", ".web.js", ".ts", ".js"]
};

const experiments = {
    syncWebAssembly: true,
    topLevelAwait: true,
};

module.exports = [
    /**
     * Notebook extension
     *
     * This bundle only contains the part of the JavaScript that is run on load of
     * the notebook.
     */
    {
        entry: './src/extension.ts',
        output: {
            filename: 'index.js',
            path: path.resolve(__dirname, 'altairfusion_jupyter', 'nbextension'),
            libraryTarget: 'amd',
            publicPath: '',
        },
        module: {
            rules: rules
        },
        mode: "production",
        devtool: 'source-map',
        externals,
        resolve,
        experiments,
    },

    /**
     * Embeddable altairfusion-jupyter bundle
     *
     * This bundle is almost identical to the notebook extension bundle. The only
     * difference is in the configuration of the webpack public path for the
     * static assets.
     *
     * The target bundle is always `dist/index.js`, which is the path required by
     * the custom widget embedder.
     */
    {
        entry: './src/index.ts',
        output: {
            filename: 'index.js',
            path: path.resolve(__dirname, 'dist'),
            libraryTarget: 'amd',
            library: "altairfusion-jupyter",
            publicPath: 'https://unpkg.com/altairfusion-jupyter@' + version + '/dist/'
        },
        devtool: 'source-map',
        module: {
            rules: rules
        },
        mode: "production",
        externals,
        resolve,
        experiments,
    },

    /**
     * Documentation widget bundle
     *
     * This bundle is used to embed widgets in the package documentation.
     */
    {
        entry: './src/index.ts',
        output: {
            filename: 'embed-bundle.js',
            path: path.resolve(__dirname, 'docs', 'source', '_static'),
            library: "altairfusion-jupyter",
            libraryTarget: 'amd'
        },
        module: {
            rules: rules
        },
        mode: "production",
        devtool: 'source-map',
        externals,
        resolve,
        experiments,
    }
];
