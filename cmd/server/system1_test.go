package main

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"parallax/internal/ai"
)

func TestModelHealth(t *testing.T) {
	var ok atomic.Bool
	ok.Store(true)
	model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" || r.Header.Get("Authorization") != "Bearer k" || !ok.Load() {
			http.Error(w, "down", http.StatusServiceUnavailable)
		}
	}))
	defer model.Close()
	h := newModelHealth(slog.New(slog.NewTextHandler(io.Discard, nil)), model.URL+"/predict", "k")

	if err := h.Check(context.Background()); err != nil || !h.Up() {
		t.Fatalf("healthy model: up=%v err=%v", h.Up(), err)
	}
	h.failed(errors.New("timeout"))
	if h.Up() {
		t.Fatal("still up after a failed request")
	}
	ok.Store(false)
	if h.Check(context.Background()) == nil || h.Up() {
		t.Fatal("a failing health check brought it back")
	}
	ok.Store(true)
	if err := h.Check(context.Background()); err != nil || !h.Up() {
		t.Fatalf("recovered model: up=%v err=%v", h.Up(), err)
	}
	model.Close()
	if h.Check(context.Background()) == nil || h.Up() {
		t.Fatal("unreachable model counted as up")
	}
}

func TestNewModelPool(t *testing.T) {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	urls := []string{"http://a/predict", "http://b/predict", "http://c/predict"}
	p, err := newModelPool(log, urls, []string{"k1", "k2", "k3"}, []string{"2", "2", "1"}, true, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	eps := p.Endpoints()
	if len(eps) != 3 || eps[2].Slots != 1 || eps[0].Model.(*ai.Client).APIKey != "k1" || eps[2].Model.(*ai.Client).APIKey != "k3" {
		t.Fatalf("endpoints not built per flag: %+v", eps)
	}
	if p, err = newModelPool(log, urls, []string{"k"}, nil, true, time.Second); err != nil || p.Endpoints()[1].Model.(*ai.Client).APIKey != "k" || p.Endpoints()[1].Slots != 2 {
		t.Fatalf("one key should apply to all, default 2 slots: %v", err)
	}
	for _, bad := range [][2][]string{{{"a", "b"}, nil}, {nil, {"1", "2"}}, {nil, {"0"}}, {nil, {"x"}}} {
		if _, err := newModelPool(log, urls, bad[0], bad[1], true, time.Second); err == nil {
			t.Errorf("keys %v slots %v: want an error", bad[0], bad[1])
		}
	}
}
