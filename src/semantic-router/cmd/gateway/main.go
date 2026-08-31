package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func main() {
	var (
		configPath = flag.String("config", "", "path to the router config.yaml (required)")
		listenAddr = flag.String("listen", ":8081", "HTTP bind address (default 8081 to avoid the router apiserver's default 8080)")
		timeout    = flag.Duration("upstream-timeout", 5*time.Minute, "upstream request timeout")
	)
	flag.Parse()

	if *configPath == "" {
		fmt.Fprintln(os.Stderr, "gateway: --config is required")
		os.Exit(2)
	}

	zlogger, err := logging.InitLoggerFromEnv()
	if err != nil {
		fmt.Fprintf(os.Stderr, "gateway: failed to init logging: %v\n", err)
		os.Exit(2)
	}
	zlogger.Sugar().Infof("gateway_starting addr=%s config=%s", *listenAddr, *configPath)

	router, err := extproc.NewOpenAIRouter(*configPath)
	if err != nil {
		zlogger.Sugar().Fatalf("gateway: failed to build router: %v", err)
	}

	gw := gateway.NewServer(router, gateway.WithTimeout(*timeout))
	if err := gw.ListenAndServeAddr(*listenAddr); err != nil {
		zlogger.Sugar().Errorf("gateway failed: %v", err)
		os.Exit(1)
	}
}
