package ai

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// fakeAsker answers after delay, or fails with err.
type fakeAsker struct {
	delay time.Duration
	err   error
	calls atomic.Int64
	gate  chan struct{} // if set, Ask blocks until it's closed
}

func (f *fakeAsker) Ask(ctx context.Context, p Prompt) (Answers, error) {
	f.calls.Add(1)
	if f.gate != nil {
		select {
		case <-f.gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if f.delay > 0 {
		time.Sleep(f.delay)
	}
	if f.err != nil {
		return nil, f.err
	}
	return Answers{"q": {Type: "noul", Noul: 1}}, nil
}

type fakeHealth struct{ up atomic.Bool }

func (h *fakeHealth) Up() bool { return h.up.Load() }
func (h *fakeHealth) Check(ctx context.Context) error {
	if h.up.Load() {
		return nil
	}
	return errors.New("down")
}

func upHealth() *fakeHealth { h := &fakeHealth{}; h.up.Store(true); return h }

func TestPoolPrefersFirst(t *testing.T) {
	a, b := &fakeAsker{}, &fakeAsker{}
	p := NewPool(time.Second, &Endpoint{Name: "a", Model: a}, &Endpoint{Name: "b", Model: b})
	for range 5 {
		if _, err := p.Ask(context.Background(), Prompt{}); err != nil {
			t.Fatal(err)
		}
	}
	if a.calls.Load() != 5 || b.calls.Load() != 0 {
		t.Fatalf("calls a=%d b=%d, want all on a", a.calls.Load(), b.calls.Load())
	}
}

func TestPoolOverflowsWhenFull(t *testing.T) {
	a := &fakeAsker{gate: make(chan struct{})}
	b := &fakeAsker{}
	p := NewPool(time.Second, &Endpoint{Name: "a", Model: a, Slots: 1}, &Endpoint{Name: "b", Model: b})
	done := make(chan error)
	go func() { _, err := p.Ask(context.Background(), Prompt{}); done <- err }()
	for a.calls.Load() == 0 {
		time.Sleep(time.Millisecond)
	}
	// a is full, so the next request goes to b.
	if _, err := p.Ask(context.Background(), Prompt{}); err != nil {
		t.Fatal(err)
	}
	if b.calls.Load() != 1 {
		t.Fatalf("b calls = %d, want 1", b.calls.Load())
	}
	close(a.gate)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestPoolSkipsDownEndpoints(t *testing.T) {
	a, b := &fakeAsker{}, &fakeAsker{}
	ha := upHealth()
	p := NewPool(time.Second, &Endpoint{Name: "a", Model: a, Health: ha}, &Endpoint{Name: "b", Model: b, Health: upHealth()})
	ha.up.Store(false)
	if _, err := p.Ask(context.Background(), Prompt{}); err != nil {
		t.Fatal(err)
	}
	if a.calls.Load() != 0 || b.calls.Load() != 1 {
		t.Fatalf("calls a=%d b=%d, want b only", a.calls.Load(), b.calls.Load())
	}
	if !p.Up() || p.Check(context.Background()) != nil {
		t.Fatal("pool with one endpoint up should be up")
	}
}

func TestPoolRetriesOnAnother(t *testing.T) {
	for _, failure := range []error{errors.New("502 Bad Gateway"), ErrBusy} {
		a, b := &fakeAsker{err: failure}, &fakeAsker{}
		var reported atomic.Int64
		p := NewPool(time.Second,
			&Endpoint{Name: "a", Model: a, OnError: func(error) { reported.Add(1) }},
			&Endpoint{Name: "b", Model: b})
		if _, err := p.Ask(context.Background(), Prompt{}); err != nil {
			t.Fatalf("%v: %v", failure, err)
		}
		if b.calls.Load() != 1 {
			t.Fatalf("%v: no retry on b", failure)
		}
		// A busy endpoint isn't an outage.
		if want := map[bool]int64{true: 0, false: 1}[errors.Is(failure, ErrBusy)]; reported.Load() != want {
			t.Fatalf("%v: OnError called %d times, want %d", failure, reported.Load(), want)
		}
	}
}

func TestPoolGivesUpAfterTwo(t *testing.T) {
	fail := errors.New("boom")
	a, b, c := &fakeAsker{err: fail}, &fakeAsker{err: fail}, &fakeAsker{}
	p := NewPool(time.Second, &Endpoint{Model: a}, &Endpoint{Model: b}, &Endpoint{Model: c})
	if _, err := p.Ask(context.Background(), Prompt{}); !errors.Is(err, fail) {
		t.Fatalf("err = %v, want the last failure", err)
	}
	if c.calls.Load() != 0 {
		t.Fatal("tried a third endpoint")
	}
}

func TestPoolNoSlot(t *testing.T) {
	a := &fakeAsker{gate: make(chan struct{})}
	defer close(a.gate)
	p := NewPool(20*time.Millisecond, &Endpoint{Model: a, Slots: 1})
	go p.Ask(context.Background(), Prompt{})
	for a.calls.Load() == 0 {
		time.Sleep(time.Millisecond)
	}
	if _, err := p.Ask(context.Background(), Prompt{}); !errors.Is(err, ErrNoSlot) {
		t.Fatalf("err = %v, want ErrNoSlot", err)
	}
}

func TestPoolWaitsForAFreeSlot(t *testing.T) {
	a := &fakeAsker{gate: make(chan struct{})}
	p := NewPool(time.Second, &Endpoint{Model: a, Slots: 1})
	go p.Ask(context.Background(), Prompt{})
	for a.calls.Load() == 0 {
		time.Sleep(time.Millisecond)
	}
	go func() { time.Sleep(20 * time.Millisecond); close(a.gate) }()
	if _, err := p.Ask(context.Background(), Prompt{}); err != nil {
		t.Fatalf("err = %v, want an answer once the slot freed", err)
	}
}

func TestPoolNoRetryPastDeadline(t *testing.T) {
	a, b := &fakeAsker{gate: make(chan struct{})}, &fakeAsker{}
	defer close(a.gate)
	p := NewPool(time.Second, &Endpoint{Model: a}, &Endpoint{Model: b})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := p.Ask(ctx, Prompt{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline", err)
	}
	if b.calls.Load() != 0 {
		t.Fatal("retried after the deadline")
	}
}
