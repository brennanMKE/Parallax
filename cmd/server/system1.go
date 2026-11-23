package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"parallax/internal/ai"
)

// modelHealth tracks whether the System 1 model answers. While it's down,
// computer players skip it and play Planner at once rather than waiting out
// a timeout every turn; a probe of its /health brings it back. Changes are
// logged once each way.
type modelHealth struct {
	log    *slog.Logger
	name   string // the endpoint's place in the pool, for logs
	url    string
	key    string
	every  time.Duration
	client *http.Client

	// changed, if set, runs when the model goes down or comes back.
	changed func()

	mu   sync.Mutex
	down bool
}

func newModelHealth(log *slog.Logger, url, key string) *modelHealth {
	return &modelHealth{log: log, name: "1", url: url, key: key, every: 15 * time.Second,
		client: &http.Client{Timeout: 5 * time.Second}}
}

// Up reports whether players should ask the model.
func (h *modelHealth) Up() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return !h.down
}

func (h *modelHealth) failed(err error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.down {
		h.down = true
		h.log.Warn("System 1 model unreachable; skipped until it's back", "endpoint", h.name, "url", h.url, "err", err)
		h.notify()
	}
}

func (h *modelHealth) notify() {
	if h.changed != nil {
		go h.changed()
	}
}

func (h *modelHealth) answered() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.down {
		h.down = false
		h.log.Info("System 1 model is back", "endpoint", h.name, "url", h.url)
		h.notify()
	}
}

// run checks the model at startup and then every few seconds, so the lobby
// offers System 1 players only while the model answers.
func (h *modelHealth) run() {
	if h.Check(context.Background()) == nil {
		h.log.Info("System 1 model is up", "endpoint", h.name, "url", h.url)
	}
	for range time.Tick(h.every) {
		h.Check(context.Background())
	}
}

// Check asks the model's /health now.
func (h *modelHealth) Check(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimSuffix(h.url, "/predict")+"/health", nil)
	if err != nil {
		return err
	}
	if h.key != "" {
		req.Header.Set("Authorization", "Bearer "+h.key)
	}
	resp, err := h.client.Do(req)
	if err == nil {
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			err = &statusError{resp.Status}
		}
	}
	if err != nil {
		h.failed(err)
		return err
	}
	h.answered()
	return nil
}

// newModelPool makes the System 1 pool from the flags: endpoints most
// preferred first, and keys and slots either one for all or one per
// endpoint.
func newModelPool(log *slog.Logger, urls, keys, slots []string, batch bool, wait time.Duration) (*ai.Pool, error) {
	pick := func(what string, list []string, i int) (string, error) {
		switch len(list) {
		case 0:
			return "", nil
		case 1:
			return list[0], nil
		case len(urls):
			return list[i], nil
		}
		return "", fmt.Errorf("%d %s for %d endpoints: give one, or one per endpoint", len(list), what, len(urls))
	}
	var eps []*ai.Endpoint
	for i, url := range urls {
		key, err := pick("keys", keys, i)
		if err != nil {
			return nil, err
		}
		n, err := pick("concurrency values", slots, i)
		if err != nil {
			return nil, err
		}
		limit := 2
		if n != "" {
			if limit, err = strconv.Atoi(n); err != nil || limit < 1 {
				return nil, fmt.Errorf("concurrency %q: want a whole number of at least 1", n)
			}
		}
		h := newModelHealth(log, url, key)
		h.name = strconv.Itoa(i + 1)
		eps = append(eps, &ai.Endpoint{
			Name:     h.name,
			Model:    &ai.Client{URL: url, APIKey: key, Batch: batch},
			Health:   h,
			OnError:  h.failed,
			OnAnswer: h.answered,
			Slots:    limit,
		})
	}
	if len(eps) == 0 {
		return nil, fmt.Errorf("no endpoints")
	}
	return ai.NewPool(wait, eps...), nil
}

// runModelHealth starts each endpoint's health checks; changed runs when
// any of them goes down or comes back.
func runModelHealth(pool *ai.Pool, changed func()) {
	for _, e := range pool.Endpoints() {
		h := e.Health.(*modelHealth)
		h.changed = changed
		go h.run()
	}
}

type statusError struct{ status string }

func (e *statusError) Error() string { return "health check: " + e.status }
