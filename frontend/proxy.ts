// Clerk proxy. All routes are public — the backend enforces the
// anonymous tier at /ask (see docs/auth_and_rate_limit_plan.md). This
// proxy only attaches Clerk's auth context so server components
// and route handlers can call `auth()`/`getToken()` when a user IS
// signed in. Sign-in is prompted by the LimitModal (next step), not by
// middleware redirects.
import { clerkMiddleware } from '@clerk/nextjs/server';

export default clerkMiddleware();

export const config = {
  matcher: [
    // Skip Next.js internals and static files.
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    // Always run on API routes.
    '/(api|trpc)(.*)',
  ],
};
