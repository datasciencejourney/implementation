window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        scroll_to_results: function(n_clicks) {
            if (n_clicks > 0) {
                const resultsSection = document.getElementById('results-section');
                if (resultsSection) {
                    resultsSection.scrollIntoView({
                        behavior: 'smooth',
                        block: 'center'
                    });
                }
            }
            return '';
        }
    }
});